/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.cli.game.training

import scala.collection.Map

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.algorithm._
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.cli.game.GAMEDriver
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.function.glm._
import com.linkedin.photon.ml.function.svm.{DistributedSmoothedHingeLossFunction, SingleNodeSmoothedHingeLossFunction}
import com.linkedin.photon.ml.io.ModelOutputMode
import com.linkedin.photon.ml.model.GAMEModel
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.optimization.DistributedOptimizationProblem
import com.linkedin.photon.ml.optimization.game.{FactoredRandomEffectOptimizationProblem, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.projector.IdentityProjection
import com.linkedin.photon.ml.sampler.{BinaryClassificationDownSampler, DefaultDownSampler}
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.util._
import com.linkedin.photon.ml.{BroadcastLike, RDDLike, SparkContextConfiguration}

/**
 * The driver class, which provides the main entry point to GAME model training
 */
final class Driver(val params: Params, val sparkContext: SparkContext, val logger: PhotonLogger)
  extends GAMEDriver(params, sparkContext, logger) {

  protected[game] val idTypeSet: Set[String] =
    params.randomEffectDataConfigurations.values.map(_.randomEffectType).toSet ++ params.getShardedEvaluatorIdTypes

  protected[game] val defaultNormalizationContext: Broadcast[NormalizationContext] =
    sparkContext.broadcast(NoNormalization())

  /**
   * Builds a GAME dataset according to input data configuration
   *
   * @param featureShardIdToFeatureMapLoader A map of feature shard id to feature map loader
   * @return The prepared GAME dataset
   */
  protected[training] def prepareGameDataSet(
    featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader]): RDD[(Long, GameDatum)] = {

    // Get the training records path
    val trainingRecordsPath = (params.trainDateRangeOpt, params.trainDateRangeDaysAgoOpt) match {
      // Specified as date range
      case (Some(trainDateRange), None) =>
        val dateRange = DateRange.fromDates(trainDateRange)
        IOUtils.getInputPathsWithinDateRange(params.trainDirs, dateRange, hadoopConfiguration, errorOnMissing = false)

      // Specified as a range of start days ago - end days ago
      case (None, Some(trainDateRangeDaysAgo)) =>
        val dateRange = DateRange.fromDaysAgo(trainDateRangeDaysAgo)
        IOUtils.getInputPathsWithinDateRange(params.trainDirs, dateRange, hadoopConfiguration, errorOnMissing = false)

      // Both types specified: illegal
      case (Some(_), Some(_)) =>
        throw new IllegalArgumentException(
          "Both trainDateRangeOpt and trainDateRangeDaysAgoOpt given. You must specify date ranges using only one " +
          "format.")

      // No range specified, just use the train dir
      case (None, None) => params.trainDirs.toSeq
    }
    logger.debug(s"Training records paths:\n${trainingRecordsPath.mkString("\n")}")

    // Determine the number of fixed effect partitions. Default to 0 if we have no fixed effects.
    val numFixedEffectPartitions = if (params.fixedEffectDataConfigurations.nonEmpty) {
      params.fixedEffectDataConfigurations.values.map(_.minNumPartitions).max
    } else {
      0
    }

    // Determine the number of random effect partitions. Default to 0 if we have no random effects.
    val numRandomEffectPartitions = if (params.randomEffectDataConfigurations.nonEmpty) {
      params.randomEffectDataConfigurations.values.map(_.numPartitions).max
    } else {
      0
    }

    val numPartitions = math.max(numFixedEffectPartitions, numRandomEffectPartitions)
    require(numPartitions > 0, "Invalid configuration: neither fixed effect nor random effect partitions specified.")

    val gameDataPartitioner = new LongHashPartitioner(numPartitions)

    val dataReader = new AvroDataReader(sparkContext)
    val data = dataReader.readMerged(
      trainingRecordsPath,
      featureShardIdToFeatureMapLoader.toMap,
      params.featureShardIdToFeatureSectionKeysMap,
      numPartitions)

    val gameDataSet = GameConverters.getGameDataSetFromDataFrame(
      data,
      params.featureShardIdToFeatureSectionKeysMap.keys.toSet,
      idTypeSet,
      isResponseRequired = true)
      .partitionBy(gameDataPartitioner)
      .setName("GAME training data")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    gameDataSet.count()
    gameDataSet
  }

  /**
   * Prepares the training dataset
   *
   * @param gameDataSet The input dataset
   * @return The training dataset
   */
  protected[training] def prepareTrainingDataSet(
    gameDataSet: RDD[(Long, GameDatum)]): Map[String, DataSet[_ <: DataSet[_]]] = {

    val fixedEffectDataSets = params.fixedEffectDataConfigurations.map {
      case (id, fixedEffectDataConfiguration) =>
        val fixedEffectDataSet = FixedEffectDataSet.buildWithConfiguration(gameDataSet, fixedEffectDataConfiguration)
          .setName(s"Fixed effect data set with id $id")
          .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
        logger.debug(s"Fixed effect data set with id $id summary:\n${fixedEffectDataSet.toSummaryString}\n")
        (id, fixedEffectDataSet)
    }

    // Prepare the per-random effect partitioner
    val randomEffectPartitionerMap = params.randomEffectDataConfigurations.map {
      case (id, randomEffectDataConfiguration) =>
        val numPartitions = randomEffectDataConfiguration.numPartitions
        val randomEffectId = randomEffectDataConfiguration.randomEffectType
        (id, RandomEffectDataSetPartitioner.generateRandomEffectDataSetPartitionerFromGameDataSet(
          numPartitions,
          randomEffectId,
          gameDataSet))
    }

    // Prepare the random effect data sets
    val randomEffectDataSets = params.randomEffectDataConfigurations.map {
      case (id, randomEffectDataConfiguration) =>
        val randomEffectPartitioner = randomEffectPartitionerMap(id)
        val rawRandomEffectDataSet = RandomEffectDataSet
          .buildWithConfiguration(gameDataSet, randomEffectDataConfiguration, randomEffectPartitioner)
          .setName(s"Random effect data set with coordinate id $id")
          .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
          .materialize()
        val projectorType = randomEffectDataConfiguration.projectorType
        val randomEffectDataSet = projectorType match {
          case IdentityProjection => rawRandomEffectDataSet
          case _ =>
            val randomEffectDataSetInProjectedSpace = RandomEffectDataSetInProjectedSpace
              .buildWithProjectorType(rawRandomEffectDataSet, projectorType)
              .setName(s"Random effect data set in projected space with coordinate id $id")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
              .materialize()
            // Only un-persist the active data and passive data, because randomEffectDataSet and
            // randomEffectDataSetInProjectedSpace share uniqueIdToRandomEffectIds and other RDDs/Broadcasts
            rawRandomEffectDataSet.activeData.unpersist()
            rawRandomEffectDataSet.passiveDataOption.foreach(_.unpersist())
            randomEffectDataSetInProjectedSpace
        }
        logger.debug(s"Random effect data set with id $id summary:\n${randomEffectDataSet.toSummaryString}\n")
        (id, randomEffectDataSet)
    }

    randomEffectPartitionerMap.foreach(_._2.unpersistBroadcast())

    fixedEffectDataSets ++ randomEffectDataSets
  }

  /**
   * Creates the training evaluator
   *
   * @param gameDataSet The input dataset
   * @return The training evaluator
   */
  protected[training] def prepareTrainingLossEvaluator(gameDataSet: RDD[(Long, GameDatum)]): Evaluator = {

    val labelAndOffsetAndWeights = gameDataSet.mapValues(gameData =>
      (gameData.response, gameData.offset, gameData.weight))
      .setName("Training label and offset and weights")
      .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)

    labelAndOffsetAndWeights.count()

    params.taskType match {
      case TaskType.LOGISTIC_REGRESSION =>
        new LogisticLossEvaluator(labelAndOffsetAndWeights)
      case TaskType.LINEAR_REGRESSION =>
        new SquaredLossEvaluator(labelAndOffsetAndWeights)
      case TaskType.POISSON_REGRESSION =>
        new PoissonLossEvaluator(labelAndOffsetAndWeights)
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        new SmoothedHingeLossEvaluator(labelAndOffsetAndWeights)
      case _ =>
        throw new UnsupportedOperationException(s"${params.taskType} is not a valid training evaluator")
    }
  }

  /**
   * Creates the validation evaluator(s)
   *
   * @param validatingDirs The input path for validating data set
   * @param featureShardIdToFeatureMapLoader A map of shard id to feature indices
   * @return The validating game data sets and the companion evaluator
   */
  protected[training] def prepareValidatingEvaluators(
      validatingDirs: Seq[String],
      featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader]): (RDD[(Long, GameDatum)], Seq[Evaluator]) = {

    // Read and parse the validating activities
    val validatingRecordsPath = (params.validateDateRangeOpt, params.validateDateRangeDaysAgoOpt) match {
      // Specified as date range
      case (Some(validateDateRange), None) =>
        val dateRange = DateRange.fromDates(validateDateRange)
        IOUtils.getInputPathsWithinDateRange(validatingDirs, dateRange, hadoopConfiguration, errorOnMissing = false)

      // Specified as a range of start days ago - end days ago
      case (None, Some(validateDateRangeDaysAgo)) =>
        val dateRange = DateRange.fromDaysAgo(validateDateRangeDaysAgo)
        IOUtils.getInputPathsWithinDateRange(validatingDirs, dateRange, hadoopConfiguration, errorOnMissing = false)

      // Both types specified: illegal
      case (Some(_), Some(_)) =>
        throw new IllegalArgumentException(
          "Both trainDateRangeOpt and trainDateRangeDaysAgoOpt given. You must specify date ranges using only one " +
          "format.")

      // No range specified, just use the train dir
      case (None, None) => validatingDirs
    }
    logger.debug(s"Validating records paths:\n${validatingRecordsPath.mkString("\n")}")

    val dataReader = new AvroDataReader(sparkContext)
    val data = dataReader.readMerged(
      validatingRecordsPath,
      featureShardIdToFeatureMapLoader.toMap,
      params.featureShardIdToFeatureSectionKeysMap,
      params.minPartitionsForValidation)

    val partitioner = new LongHashPartitioner(data.rdd.partitions.length)

    val gameDataSet = GameConverters.getGameDataSetFromDataFrame(
      data,
      params.featureShardIdToFeatureSectionKeysMap.keys.toSet,
      idTypeSet,
      isResponseRequired = true)
      .partitionBy(partitioner)
      .setName("Validating Game data set")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    // Log some simple summary info on the Game data set
    logger.debug(s"Summary for the validating Game data set")
    val numSamples = gameDataSet.count()
    logger.debug(s"numSamples: $numSamples")
    val responseSum = gameDataSet.values.map(_.response).sum()
    logger.debug(s"responseSum: $responseSum")
    val weightSum = gameDataSet.values.map(_.weight).sum()
    logger.debug(s"weightSum: $weightSum")
    val randomEffectTypeToIdMap = gameDataSet.values.first().idTypeToValueMap
    randomEffectTypeToIdMap.keySet.foreach { randomEffectType =>
      val dataStats = gameDataSet.values.map { gameData =>
        val randomEffectId = gameData.idTypeToValueMap(randomEffectType)
        (randomEffectId, (gameData.response, 1))
      }.reduceByKey { case ((responseSum1, numSample1), (responseSum2, numSample2)) =>
        (responseSum1 + responseSum2, numSample1 + numSample2)
      }.cache()
      val responseSumStats = dataStats.values.map(_._1).stats()
      val numSamplesStats = dataStats.values.map(_._2).stats()
      logger.debug(s"numSamplesStats for $randomEffectType: $numSamplesStats")
      logger.debug(s"responseSumStats for $randomEffectType: $responseSumStats")
    }

    val validatingLabelsAndOffsetsAndWeights = gameDataSet
      .mapValues(gameData => (gameData.response, gameData.offset, gameData.weight))
      .setName(s"Validating labels and offsets").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    validatingLabelsAndOffsetsAndWeights.count()

    val evaluators =
      if (params.evaluatorTypes.isEmpty) {
        // Get default evaluators given the task type
        val defaultEvaluator =
          params.taskType match {
            case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
              new AreaUnderROCCurveEvaluator(validatingLabelsAndOffsetsAndWeights)
            case TaskType.LINEAR_REGRESSION =>
              new RMSEEvaluator(validatingLabelsAndOffsetsAndWeights)
            case TaskType.POISSON_REGRESSION =>
              new PoissonLossEvaluator(validatingLabelsAndOffsetsAndWeights)
            case _ =>
              throw new UnsupportedOperationException(s"${params.taskType} is not a valid validating evaluator")
          }
        Seq(defaultEvaluator)
      } else {
        params.evaluatorTypes.map(Evaluator.buildEvaluator(_, gameDataSet))
      }
    val randomScores = gameDataSet.mapValues(_ => math.random)
    evaluators.foreach { evaluator =>
      val metric = evaluator.evaluate(randomScores)
      logger.info(s"Random guessing based baseline evaluation metric for ${evaluator.getEvaluatorName}: $metric")
    }
    (gameDataSet, evaluators)
  }

  /**
   * Train GAME models. This method builds a coordinate descent optimization problem from the individual optimization
   * problems for the fixed effect, random effect, and factored random effect models.
   *
   * @param dataSets The training datasets
   * @param trainingEvaluator The training evaluator
   * @param validatingDataAndEvaluatorsOption Optional validation dataset and evaluators
   * @return Trained GAME model
   */
  protected[training] def train(
      dataSets: Map[String, DataSet[_ <: DataSet[_]]],
      trainingEvaluator: Evaluator,
      validatingDataAndEvaluatorsOption: Option[(RDD[(Long, GameDatum)], Seq[Evaluator])]): Map[String, GAMEModel] = {

    val gameModels = for (
        fixedEffectOptimizationConfiguration <- params.fixedEffectOptimizationConfigurations;
        randomEffectOptimizationConfiguration <- params.randomEffectOptimizationConfigurations;
        factoredRandomEffectOptimizationConfiguration <- params.factoredRandomEffectOptimizationConfigurations) yield {

      // TODO: this is only geared toward readability by humans, but we need to save those in machine readable format
      // as model metadata too
      val modelConfig = fixedEffectOptimizationConfiguration.mkString("\n") + "\n" +
          randomEffectOptimizationConfiguration.mkString("\n") + "\n" +
          factoredRandomEffectOptimizationConfiguration.mkString("\n")

      val timer = Timer.start()
      logger.info(s"Start to train the game model with the following config:\n$modelConfig\n")

      val glmConstructor = params.taskType match {
        case TaskType.LOGISTIC_REGRESSION => LogisticRegressionModel.apply _
        case TaskType.LINEAR_REGRESSION => LinearRegressionModel.apply _
        case TaskType.POISSON_REGRESSION => PoissonRegressionModel.apply _
        case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => SmoothedHingeLossLinearSVMModel.apply _
        case _ => throw new Exception("Need to specify a valid loss function")
      }

      // For each model, create optimization coordinates for the fixed effect, random effect, and factored random effect
      // models
      val coordinates = params.updatingSequence.map { coordinateId =>
        val coordinate = dataSets(coordinateId) match {
          // Fixed effect coordinate
          case fixedEffectDataSet: FixedEffectDataSet =>
            val optimizationConfiguration = fixedEffectOptimizationConfiguration(coordinateId)
            val downSamplingRate = optimizationConfiguration.downSamplingRate
            // If number of features is from moderate to large (>200000), then use tree aggregate,
            // otherwise use aggregate.
            val treeAggregateDepth = if (fixedEffectDataSet.numFeatures < Driver.FIXED_EFFECT_FEATURE_THRESHOLD) {
              Driver.DEFAULT_TREE_AGGREGATE_DEPTH
            } else {
              Driver.DEEP_TREE_AGGREGATE_DEPTH
            }
            val objectiveFunction = params.taskType match {
              case TaskType.LOGISTIC_REGRESSION =>
                DistributedGLMLossFunction.create(
                  optimizationConfiguration,
                  LogisticLossFunction,
                  sparkContext,
                  treeAggregateDepth)

              case TaskType.LINEAR_REGRESSION =>
                DistributedGLMLossFunction.create(
                  optimizationConfiguration,
                  SquaredLossFunction,
                  sparkContext,
                  treeAggregateDepth)

              case TaskType.POISSON_REGRESSION =>
                DistributedGLMLossFunction.create(
                  optimizationConfiguration,
                  PoissonLossFunction,
                  sparkContext,
                  treeAggregateDepth)

              case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
                DistributedSmoothedHingeLossFunction.create(
                  optimizationConfiguration,
                  sparkContext,
                  treeAggregateDepth)
            }
            val samplerOption = if (downSamplingRate > 0D && downSamplingRate < 1D) {
              params.taskType match {
                case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
                  Some(new BinaryClassificationDownSampler(downSamplingRate))
                case TaskType.LINEAR_REGRESSION | TaskType.POISSON_REGRESSION =>
                  Some(new DefaultDownSampler(downSamplingRate))
              }
            } else {
              None
            }
            val optimizationProblem = DistributedOptimizationProblem.create(
              optimizationConfiguration,
              objectiveFunction,
              samplerOption,
              glmConstructor,
              defaultNormalizationContext,
              Driver.TRACK_STATE,
              params.computeVariance
            )

            new FixedEffectCoordinate(fixedEffectDataSet, optimizationProblem)

          case randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace =>
            // Random effect coordinate
            val optimizationConfiguration = randomEffectOptimizationConfiguration(coordinateId)
            val objectiveFunction = params.taskType match {
              case TaskType.LOGISTIC_REGRESSION =>
                SingleNodeGLMLossFunction.create(
                  optimizationConfiguration,
                  LogisticLossFunction)

              case TaskType.LINEAR_REGRESSION =>
                SingleNodeGLMLossFunction.create(
                  optimizationConfiguration,
                  SquaredLossFunction)

              case TaskType.POISSON_REGRESSION =>
                SingleNodeGLMLossFunction.create(
                  optimizationConfiguration,
                  PoissonLossFunction)

              case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
                SingleNodeSmoothedHingeLossFunction.create(
                  optimizationConfiguration)
            }
            val randomEffectOptimizationProblem = RandomEffectOptimizationProblem
              .create(
                randomEffectDataSetInProjectedSpace,
                optimizationConfiguration,
                objectiveFunction,
                glmConstructor,
                defaultNormalizationContext,
                Driver.TRACK_STATE,
                params.computeVariance)
              .setName(s"Random effect optimization problem of coordinate $coordinateId")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

            new RandomEffectCoordinateInProjectedSpace(
              randomEffectDataSetInProjectedSpace,
              randomEffectOptimizationProblem)

          case randomEffectDataSet: RandomEffectDataSet =>
            // Factored random effect coordinate
            val (randomEffectOptimizationConfiguration,
              latentFactorOptimizationConfiguration,
              mfOptimizationConfiguration) = factoredRandomEffectOptimizationConfiguration(coordinateId)

            val downSamplingRate = latentFactorOptimizationConfiguration.downSamplingRate
            val samplerOption = if (downSamplingRate > 0D && downSamplingRate < 1D) {
              params.taskType match {
                case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
                  Some(new BinaryClassificationDownSampler(downSamplingRate))
                case TaskType.LINEAR_REGRESSION | TaskType.POISSON_REGRESSION =>
                  Some(new DefaultDownSampler(downSamplingRate))
              }
            } else {
              None
            }
            val (randomObjectiveFunction, latentObjectiveFunction) = params.taskType match {
              case TaskType.LOGISTIC_REGRESSION =>
                val random = SingleNodeGLMLossFunction.create(
                  randomEffectOptimizationConfiguration,
                  LogisticLossFunction)
                val latent = DistributedGLMLossFunction.create(
                  latentFactorOptimizationConfiguration,
                  LogisticLossFunction,
                  sparkContext,
                  Driver.DEFAULT_TREE_AGGREGATE_DEPTH)
                (random, latent)

              case TaskType.LINEAR_REGRESSION =>
                val random = SingleNodeGLMLossFunction.create(
                  randomEffectOptimizationConfiguration,
                  SquaredLossFunction)
                val latent = DistributedGLMLossFunction.create(
                  latentFactorOptimizationConfiguration,
                  SquaredLossFunction,
                  sparkContext,
                  Driver.DEFAULT_TREE_AGGREGATE_DEPTH)
                (random, latent)

              case TaskType.POISSON_REGRESSION =>
                val random = SingleNodeGLMLossFunction.create(
                  randomEffectOptimizationConfiguration,
                  PoissonLossFunction)
                val latent = DistributedGLMLossFunction.create(
                  latentFactorOptimizationConfiguration,
                  PoissonLossFunction,
                  sparkContext,
                  Driver.DEFAULT_TREE_AGGREGATE_DEPTH)
                (random, latent)

              case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
                val random = SingleNodeSmoothedHingeLossFunction.create(
                  randomEffectOptimizationConfiguration)
                val latent = DistributedSmoothedHingeLossFunction.create(
                  latentFactorOptimizationConfiguration,
                  sparkContext,
                  Driver.DEFAULT_TREE_AGGREGATE_DEPTH)
                (random, latent)
            }
            val factoredRandomEffectOptimizationProblem = FactoredRandomEffectOptimizationProblem
              .create(
                randomEffectDataSet,
                randomEffectOptimizationConfiguration,
                latentFactorOptimizationConfiguration,
                mfOptimizationConfiguration,
                randomObjectiveFunction,
                latentObjectiveFunction,
                samplerOption,
                glmConstructor,
                defaultNormalizationContext,
                Driver.TRACK_STATE,
                params.computeVariance)
              .setName(s"Factored random effect optimization problem of coordinate $coordinateId")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

            new FactoredRandomEffectCoordinate(randomEffectDataSet, factoredRandomEffectOptimizationProblem)

          case dataSet =>
            throw new UnsupportedOperationException(s"Data set of type ${dataSet.getClass} is not supported")
        }
        Pair[String, Coordinate[_ <: DataSet[_], _ <: Coordinate[_, _]]](coordinateId, coordinate)
      }
      val coordinateDescent =
        new CoordinateDescent(coordinates, trainingEvaluator, validatingDataAndEvaluatorsOption, logger)
      val gameModel = coordinateDescent.run(params.numIterations, params.taskType)

      timer.stop()
      logger.info(s"Finished training model with the following config:\n$modelConfig\n" +
          s"Time elapsed: ${timer.durationSeconds} (s)\n")

      (modelConfig, gameModel)
    }

    gameModels.toMap
  }

  /**
   * Select best model according to validation evaluator
   *
   * TODO: possibly remove the String from the output, in favor of a method that parse the model metadata?
   *
   * @param gameModels the models to evaluate (single evaluator, on the validation data set)
   * @param validationDataAndEvaluatorsOption only the first evaluator is used!
   * @return the best model
   */
  protected[training] def selectBestModel(
    gameModels : Map[String, GAMEModel],
    validationDataAndEvaluatorsOption: Option[(RDD[(Long, GameDatum)], Seq[Evaluator])])
  : Option[(String, GAMEModel, Double)] = {

    validationDataAndEvaluatorsOption match {

      case Some((data, evaluators)) =>

        val evaluator = evaluators.head
        logger.debug(s"Selecting best model according to evaluator ${evaluator.getEvaluatorName}")

        val (bestConfig, bestModel: GAMEModel, bestScore) = gameModels
          .map { case (modelConfig, model) => (modelConfig, model, evaluator.evaluate(model.score(data).scores)) }
          .reduce { (configModelScore1, configModelScore2) =>
            if (evaluator.betterThan(configModelScore1._3, configModelScore2._3))
              { configModelScore1 } else { configModelScore2 }
          }

        logger.info(s"The selected model has the following config:\n$bestConfig\nModel summary:" +
          s"\n${bestModel.toSummaryString}\n\nEvaluation result is : $bestScore")

        Some((bestConfig, bestModel, bestScore))

      case None =>
        logger.debug("No best model selection because no validation data was provided")
        None
    }
  }

  /**
   * Write the GAME models to HDFS
   *
   * TODO: deprecate model-spec then remove it in favor of model-metadata, but there are clients!
   *
   * @param featureShardIdToFeatureMapLoader the shard ids
   * @param gameModelsMap all the models that were producing during training
   * @param bestModel the best model
   */
  protected[training] def saveModelToHDFS(
    featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader],
    gameModelsMap: Map[String, GAMEModel],
    bestModel: Option[(String, GAMEModel, Double)]) {

    // Write the best model to HDFS
    bestModel match {

      case Some((modelConfig, model, _)) =>

        val modelOutputDir = new Path(params.outputDir, "best").toString
        Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)

        // TODO: deprecate this
        val modelSpecDir = new Path(modelOutputDir, "model-spec").toString
        IOUtils.writeStringsToHDFS(Iterator(modelConfig), modelSpecDir, hadoopConfiguration,
          forceOverwrite = false)

        ModelProcessingUtils.saveGameModelsToHDFS(model, featureShardIdToFeatureMapLoader, modelOutputDir,
          params, sparkContext)

        logger.info("Saved model to HDFS")

      case None =>
        logger.info("No model to save to HDFS")
    }

    // Write all models to HDFS
    // TODO: just output the best model once we have hyperparameter optimization
    if (params.modelOutputMode == ModelOutputMode.ALL) {
      gameModelsMap.foldLeft(0) { case (modelIndex, (modelConfig, gameModel)) =>
        val modelOutputDir = new Path(params.outputDir, s"all/$modelIndex").toString

        // TODO: deprecate this
        val modelSpecDir = new Path(modelOutputDir, "model-spec").toString

        Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)
        IOUtils.writeStringsToHDFS(Iterator(modelConfig), modelSpecDir, hadoopConfiguration, forceOverwrite = false)
        ModelProcessingUtils.saveGameModelsToHDFS(
          gameModel,
          featureShardIdToFeatureMapLoader,
          modelOutputDir,
          params,
          sparkContext)

        modelIndex + 1
      }
    }
  }

  /**
   * Run the driver
   */
  def run(): Unit = {
    val timer = new Timer

    // Process the output directory upfront and potentially fail the job early
    IOUtils.processOutputDir(params.outputDir, params.deleteOutputDirIfExists, sparkContext.hadoopConfiguration)

    timer.start()
    val featureShardIdToFeatureMapMap = prepareFeatureMaps()
    timer.stop()
    logger.info(s"Time elapsed after preparing feature maps: ${timer.durationSeconds} (s)\n")

    timer.start()
    val gameDataSet = prepareGameDataSet(featureShardIdToFeatureMapMap)
    timer.stop()
    logger.info(s"Time elapsed after game data set preparation: ${timer.durationSeconds} (s)\n")

    timer.start()
    val trainingDataSet = prepareTrainingDataSet(gameDataSet)
    timer.stop()
    logger.info(s"Time elapsed after training data set preparation: ${timer.durationSeconds} (s)\n")

    timer.start()
    val trainingLossFunctionEvaluator = prepareTrainingLossEvaluator(gameDataSet)
    timer.stop()
    logger.info(s"Time elapsed after training evaluator preparation: ${timer.durationSeconds} (s)\n")

    // Get rid of the largest object, which is no longer needed in the following code
    gameDataSet.unpersist()

    val validatingDataAndEvaluatorsOption = params.validateDirsOpt match {
      case Some(validatingDirs) =>
        timer.start()
        val validatingDataAndEvaluators = prepareValidatingEvaluators(validatingDirs, featureShardIdToFeatureMapMap)
        timer.stop()
        logger.info("Time elapsed after validating data and evaluator preparation: " +
                    s"${timer.durationSeconds} (s)\n")
        Some(validatingDataAndEvaluators)
      case None =>
        None
    }

    timer.start()
    val gameModelsMap = train(trainingDataSet, trainingLossFunctionEvaluator, validatingDataAndEvaluatorsOption)
    timer.stop()
    logger.info(s"Time elapsed after game model training: ${timer.durationSeconds} (s)\n")

    trainingDataSet.foreach { (pair: (String, DataSet[_])) =>
      val dataset = pair._2

      dataset match {
        case rddLike: RDDLike => rddLike.unpersistRDD()
        case _ =>
      }
      dataset match {
        case broadcastLike: BroadcastLike => broadcastLike.unpersistBroadcast()
        case _ =>
      }
    }

    timer.start()
    val bestModel = selectBestModel(gameModelsMap, validatingDataAndEvaluatorsOption)
    timer.stop()
    logger.info(s"Time elapsed after selecting best model: ${timer.durationSeconds} (s)\n")

    if (params.modelOutputMode != ModelOutputMode.NONE) {
      timer.start()
      saveModelToHDFS(featureShardIdToFeatureMapMap, gameModelsMap, bestModel)
      timer.stop()
      logger.info(s"Time elapsed after saving game models to HDFS: ${timer.durationSeconds} (s)\n")
    }
  }
}

object Driver {

  val FIXED_EFFECT_FEATURE_THRESHOLD = 200000
  val DEFAULT_TREE_AGGREGATE_DEPTH = 1
  val DEEP_TREE_AGGREGATE_DEPTH = 2
  val TRACK_STATE = true
  val LOGS = "logs"

  /**
   * Main entry point
   */
  def main(args: Array[String]): Unit = {

    val timer = Timer.start()

    val params = Params.parseFromCommandLine(args)
    import params._

    val sc = SparkContextConfiguration.asYarnClient(applicationName, useKryo = true)

    val logsDir = new Path(outputDir, LOGS).toString
    val logger = new PhotonLogger(logsDir, sc)
    //TODO: This Photon log level should be made configurable
    logger.setLogLevel(PhotonLogger.LogLevelDebug)

    try {
      logger.debug(params.toString + "\n")

      val job = new Driver(params, sc, logger)
      job.run()

      timer.stop()
      logger.info(s"Overall time elapsed ${timer.durationMinutes} minutes")
    } catch {
      case e: Exception =>
        logger.error("Failure while running the driver", e)
        throw e
    } finally {
      logger.close()
      sc.stop()
    }
  }
}
