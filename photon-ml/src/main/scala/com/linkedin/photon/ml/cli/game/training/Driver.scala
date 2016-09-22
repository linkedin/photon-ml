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
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.cli.game.GAMEDriver
import com.linkedin.photon.ml.algorithm._
import com.linkedin.photon.ml.avro.AvroUtils
import com.linkedin.photon.ml.avro.data.DataProcessingUtils
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.io.ModelOutputMode
import com.linkedin.photon.ml.model.GAMEModel
import com.linkedin.photon.ml.optimization.game.{GLMOptimizationConfiguration, FactoredRandomEffectOptimizationProblem,
  RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.optimization.{GeneralizedLinearOptimizationProblem, LinearRegressionOptimizationProblem,
  LogisticRegressionOptimizationProblem, PoissonRegressionOptimizationProblem,
  SmoothedHingeLossLinearSVMOptimizationProblem}
import com.linkedin.photon.ml.projector.IdentityProjection
import com.linkedin.photon.ml.supervised.TaskType._
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util._
import com.linkedin.photon.ml.{RDDLike, SparkContextConfiguration}

/**
  * The driver class, which provides the main entrance to GAME model training
  */
final class Driver(val params: Params, val sparkContext: SparkContext, val logger: PhotonLogger)
    extends GAMEDriver(params, sparkContext, logger) {

  import params._

  protected[game] val idTypeSet: Set[String] = {
    val randomEffectIdTypeSet = randomEffectDataConfigurations.values.map(_.randomEffectId).toSet
    randomEffectIdTypeSet ++ getPrecisionAtKIdTypeSet
  }

  /**
   * Builds a GAME dataset according to input data configuration
   *
   * @param featureShardIdToFeatureMapLoader A map of feature shard id to feature map loader
   * @return The prepared GAME dataset
   */
  protected[training] def prepareGameDataSet(featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader])
    : RDD[(Long, GameDatum)] = {

    // Get the training records path
    val trainingRecordsPath = (trainDateRangeOpt, trainDateRangeDaysAgoOpt) match {
      // Specified as date range
      case (Some(trainDateRange), None) =>
        val dateRange = DateRange.fromDates(trainDateRange)
        IOUtils.getInputPathsWithinDateRange(trainDirs, dateRange, hadoopConfiguration, errorOnMissing = false)

      // Specified as a range of start days ago - end days ago
      case (None, Some(trainDateRangeDaysAgo)) =>
        val dateRange = DateRange.fromDaysAgo(trainDateRangeDaysAgo)
        IOUtils.getInputPathsWithinDateRange(trainDirs, dateRange, hadoopConfiguration, errorOnMissing = false)

      // Both types specified: illegal
      case (Some(_), Some(_)) =>
        throw new IllegalArgumentException(
          "Both trainDateRangeOpt and trainDateRangeDaysAgoOpt given. You must specify date ranges using only one " +
          "format.")

      // No range specified, just use the train dir
      case (None, None) => trainDirs.toSeq
    }
    logger.debug(s"Training records paths:\n${trainingRecordsPath.mkString("\n")}")

    // Determine the number of fixed effect partitions. Default to 0 if we have no fixed effects.
    val numFixedEffectPartitions = if (fixedEffectDataConfigurations.nonEmpty) {
      fixedEffectDataConfigurations.values.map(_.minNumPartitions).max
    } else {
      0
    }

    // Determine the number of random effect partitions. Default to 0 if we have no random effects.
    val numRandomEffectPartitions = if (randomEffectDataConfigurations.nonEmpty) {
      randomEffectDataConfigurations.values.map(_.numPartitions).max
    } else {
      0
    }

    val numPartitions = math.max(numFixedEffectPartitions, numRandomEffectPartitions)
    require(numPartitions > 0, "Invalid configuration: neither fixed effect nor random effect partitions specified.")

    val records = AvroUtils.readAvroFiles(sparkContext, trainingRecordsPath, numPartitions)
    val recordsWithUniqueId = records.zipWithUniqueId().map(_.swap)
    val globalDataPartitioner = new LongHashPartitioner(records.partitions.length)

    val gameDataSet = DataProcessingUtils.getGameDataSetFromGenericRecords(
      recordsWithUniqueId,
      featureShardIdToFeatureSectionKeysMap,
      featureShardIdToFeatureMapLoader,
      idTypeSet,
      isResponseRequired = true)
      .partitionBy(globalDataPartitioner)
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
  protected[training] def prepareTrainingDataSet(gameDataSet: RDD[(Long, GameDatum)])
    : Map[String, DataSet[_ <: DataSet[_]]] = {

    val fixedEffectDataSets = fixedEffectDataConfigurations.map { case (id, fixedEffectDataConfiguration) =>
      val fixedEffectDataSet = FixedEffectDataSet.buildWithConfiguration(gameDataSet, fixedEffectDataConfiguration)
        .setName(s"Fixed effect data set with id $id")
        .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
      logger.debug(s"Fixed effect data set with id $id summary:\n${fixedEffectDataSet.toSummaryString}\n")
      (id, fixedEffectDataSet)
    }

    // Prepare the per-random effect partitioner
    val randomEffectPartitionerMap = randomEffectDataConfigurations.map { case (id, randomEffectDataConfiguration) =>
      val numPartitions = randomEffectDataConfiguration.numPartitions
      val randomEffectId = randomEffectDataConfiguration.randomEffectId
      (id, RandomEffectIdPartitioner.generateRandomEffectIdPartitionerFromGameDataSet(numPartitions,
        randomEffectId, gameDataSet))
    }

    // Prepare the random effect data sets
    val randomEffectDataSet = randomEffectDataConfigurations.map { case (id, randomEffectDataConfiguration) =>
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
          // randomEffectDataSetInProjectedSpace share globalIdToIndividualIds and other RDDs/Broadcasts
          rawRandomEffectDataSet.activeData.unpersist()
          rawRandomEffectDataSet.passiveDataOption.foreach(_.unpersist())
          randomEffectDataSetInProjectedSpace
      }
      logger.debug(s"Random effect data set with id $id summary:\n${randomEffectDataSet.toSummaryString}\n")
      (id, randomEffectDataSet)
    }

    fixedEffectDataSets ++ randomEffectDataSet
  }

  /**
   * Creates the training evaluator
   *
   * @param gameDataSet The input dataset
   * @return The training evaluator
   */
  protected[training] def prepareTrainingLossFunctionEvaluator(gameDataSet: RDD[(Long, GameDatum)]): Evaluator = {
    val labelAndOffsetAndWeights = gameDataSet.mapValues(gameData =>
      (gameData.response, gameData.offset, gameData.weight)
    ).setName("Training label and offset and weights").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    labelAndOffsetAndWeights.count()
    taskType match {
      case LOGISTIC_REGRESSION =>
        new LogisticLossEvaluator(labelAndOffsetAndWeights)
      case LINEAR_REGRESSION =>
        new SquaredLossEvaluator(labelAndOffsetAndWeights)
      case POISSON_REGRESSION =>
        new PoissonLossEvaluator(labelAndOffsetAndWeights)
      case SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        new SmoothedHingeLossEvaluator(labelAndOffsetAndWeights)
      case _ =>
        throw new UnsupportedOperationException(s"Task type: $taskType is not supported to create training evaluator")
    }
  }

  /**
   * Creates the validation evaluator
   *
   * @param validatingDirs The input path for validating data set
   * @return The validating game data sets and the companion evaluator
   */
  protected[training] def prepareValidatingEvaluators(
      validatingDirs: Seq[String],
      featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader]): (RDD[(Long, GameDatum)], Seq[Evaluator]) = {

    // Read and parse the validating activities
    val validatingRecordsPath = (validateDateRangeOpt, validateDateRangeDaysAgoOpt) match {
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

    val records = AvroUtils.readAvroFiles(sparkContext, validatingRecordsPath, minPartitionsForValidation)
    val recordsWithUniqueId = records.zipWithUniqueId().map(_.swap)
    val partitioner = new LongHashPartitioner(records.partitions.length)

    val gameDataSet = DataProcessingUtils.getGameDataSetFromGenericRecords(
      recordsWithUniqueId,
      featureShardIdToFeatureSectionKeysMap,
      featureShardIdToFeatureMapLoader,
      idTypeSet,
      isResponseRequired = true)
      .partitionBy(partitioner).setName("Validating Game data set")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    // Log some simple summary info on the Game data set
    logger.debug(s"Summary for the validating Game data set")
    val numSamples = gameDataSet.count()
    logger.debug(s"numSamples: $numSamples")
    val responseSum = gameDataSet.values.map(_.response).sum()
    logger.debug(s"responseSum: $responseSum")
    val weightSum = gameDataSet.values.map(_.weight).sum()
    logger.debug(s"weightSum: $weightSum")
    val randomEffectIdToIndividualIdMap = gameDataSet.values.first().idTypeToValueMap
    randomEffectIdToIndividualIdMap.keySet.foreach { randomEffectId =>
      val dataStats = gameDataSet.values.map { gameData =>
        val individualId = gameData.idTypeToValueMap(randomEffectId)
        (individualId, (gameData.response, 1))
      }.reduceByKey { case ((responseSum1, numSample1), (responseSum2, numSample2)) =>
        (responseSum1 + responseSum2, numSample1 + numSample2)
      }.cache()
      val responseSumStats = dataStats.values.map(_._1).stats()
      val numSamplesStats = dataStats.values.map(_._2).stats()
      logger.debug(s"numSamplesStats for $randomEffectId: $numSamplesStats")
      logger.debug(s"responseSumStats for $randomEffectId: $responseSumStats")
    }

    val validatingLabelsAndOffsetsAndWeights = gameDataSet
      .mapValues(gameData => (gameData.response, gameData.offset, gameData.weight))
      .setName(s"Validating labels and offsets").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    validatingLabelsAndOffsetsAndWeights.count()

    val evaluators =
      if (evaluatorTypes.isEmpty) {
        // Get default evaluators given the task type
        val defaultEvaluator =
          taskType match {
            case LOGISTIC_REGRESSION | SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
              new AreaUnderROCCurveEvaluator(validatingLabelsAndOffsetsAndWeights)
            case LINEAR_REGRESSION =>
              new RMSEEvaluator(validatingLabelsAndOffsetsAndWeights)
            case POISSON_REGRESSION =>
              new PoissonLossEvaluator(validatingLabelsAndOffsetsAndWeights)
            case _ =>
              throw new UnsupportedOperationException(s"Task type: $taskType is not supported to create validating " +
                  s"evaluator")
          }
        Seq(defaultEvaluator)
      } else {
        evaluatorTypes.map(Evaluator.buildEvaluator(_, gameDataSet))
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

    val optimizationProblemBuilder: (GLMOptimizationConfiguration, Int, Boolean, Boolean) =>
        GeneralizedLinearOptimizationProblem[GeneralizedLinearModel, DiffFunction[LabeledPoint]] = taskType match {

      case LOGISTIC_REGRESSION => LogisticRegressionOptimizationProblem.buildOptimizationProblem
      case LINEAR_REGRESSION => LinearRegressionOptimizationProblem.buildOptimizationProblem
      case POISSON_REGRESSION => PoissonRegressionOptimizationProblem.buildOptimizationProblem
      case SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        SmoothedHingeLossLinearSVMOptimizationProblem.buildOptimizationProblem

      case _ => throw new Exception(s"Loss function for taskType $taskType is currently not supported.")
    }

    val gameModels = for (
        fixedEffectOptimizationConfiguration <- fixedEffectOptimizationConfigurations;
        randomEffectOptimizationConfiguration <- randomEffectOptimizationConfigurations;
        factoredRandomEffectOptimizationConfiguration <- factoredRandomEffectOptimizationConfigurations) yield {

      val modelConfig = fixedEffectOptimizationConfiguration.mkString("\n") + "\n" +
          randomEffectOptimizationConfiguration.mkString("\n") + "\n" +
          factoredRandomEffectOptimizationConfiguration.mkString("\n")

      val timer = Timer.start()
      logger.info(s"Start to train the game model with the following config:\n$modelConfig\n")

      // For each model, create optimization coordinates for the fixed effect, random effect, and factored random effect
      // models
      val coordinates = updatingSequence.map { coordinateId =>
        val coordinate = dataSets(coordinateId) match {
          case fixedEffectDataSet: FixedEffectDataSet =>
            // Fixed effect coordinate
            val optimizationConfiguration = fixedEffectOptimizationConfiguration(coordinateId)
            // If number of features is from moderate to large (>200000), then use tree aggregate,
            // otherwise use aggregate.
            val treeAggregateDepth = if (fixedEffectDataSet.numFeatures < 200000) {
                Driver.DEFAULT_TREE_AGGREGATE_DEPTH
              } else {
                Driver.DEEP_TREE_AGGREGATE_DEPTH
              }
            val optimizationProblem = optimizationProblemBuilder(
              optimizationConfiguration,
              treeAggregateDepth,
              Driver.TRACK_STATE,
              computeVariance)
            new FixedEffectCoordinate(fixedEffectDataSet, optimizationProblem)

          case randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace =>
            // Random effect coordinate
            val optimizationConfiguration = randomEffectOptimizationConfiguration(coordinateId)
            val randomEffectOptimizationProblem = RandomEffectOptimizationProblem
              .buildRandomEffectOptimizationProblem(
                optimizationProblemBuilder,
                optimizationConfiguration,
                randomEffectDataSetInProjectedSpace,
                Driver.DEFAULT_TREE_AGGREGATE_DEPTH,
                computeVariance)
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
            val factoredRandomEffectOptimizationProblem = FactoredRandomEffectOptimizationProblem
              .buildFactoredRandomEffectOptimizationProblem(
                optimizationProblemBuilder,
                randomEffectOptimizationConfiguration,
                latentFactorOptimizationConfiguration,
                mfOptimizationConfiguration,
                randomEffectDataSet,
                Driver.DEFAULT_TREE_AGGREGATE_DEPTH,
                Driver.TRACK_STATE,
                computeVariance)
              .setName(s"Factored random effect optimization problem of coordinate $coordinateId")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
            new FactoredRandomEffectCoordinate(randomEffectDataSet, factoredRandomEffectOptimizationProblem)

          case dataSet =>
            throw new UnsupportedOperationException(s"Data set of type ${dataSet.getClass} is not supported")
        }
        Pair[String, Coordinate[_ <: DataSet[_], _ <: Coordinate[_, _]]](coordinateId, coordinate)
      }
      val coordinateDescent = new CoordinateDescent(coordinates, trainingEvaluator, validatingDataAndEvaluatorsOption,
        logger)
      val gameModel = coordinateDescent.run(numIterations)

      timer.stop()
      logger.info(s"Finished training model with the following config:\n$modelConfig\n" +
          s"Time elapsed: ${timer.durationSeconds} (s)\n")

      (modelConfig, gameModel)
    }

    gameModels.toMap
  }

  /**
   * Write the learned GAME model to HDFS
   *
   * @param featureShardIdToFeatureMapLoader A map of feature shard id to feature map loader
   * @param validatingDataAndEvaluatorOption Optional validation dataset and evaluator
   * @param gameModelsMap GAME models
   */
  protected[training] def saveModelToHDFS(
      featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader],
      validatingDataAndEvaluatorOption: Option[(RDD[(Long, GameDatum)], Evaluator)],
      gameModelsMap: Map[String, GAMEModel]) {

    // Write the best model to HDFS
    validatingDataAndEvaluatorOption match {
      case Some((validatingData, evaluator)) =>
        val (bestModelConfig, evaluationResult) = gameModelsMap.mapValues(_.score(validatingData).scores)
          .mapValues(evaluator.evaluate)
          .reduce((result1, result2) => if (evaluator.betterThan(result1._2, result2._2)) result1 else result2)

        val bestGameModel = gameModelsMap.get(bestModelConfig).get
        logger.info(s"The selected model has the following config:\n$bestModelConfig\nModel summary:" +
          s"\n${bestGameModel.toSummaryString}\n\nEvaluation result is : $evaluationResult")

        val modelOutputDir = new Path(outputDir, "best").toString
        Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)
        val modelSpecDir = new Path(modelOutputDir, "model-spec").toString
        IOUtils.writeStringsToHDFS(Iterator(bestModelConfig), modelSpecDir, hadoopConfiguration, forceOverwrite = false)
        ModelProcessingUtils.saveGameModelsToHDFS(bestGameModel, featureShardIdToFeatureMapLoader, modelOutputDir,
          numberOfOutputFilesForRandomEffectModel, sparkContext)
      case _ =>
        logger.info("No validation data provided: cannot determine best model, thus no 'best model' output.")
    }

    // Write all models to HDFS
    if (modelOutputMode == ModelOutputMode.ALL) {
      var modelIdx = 0
      gameModelsMap.foreach { case (modelConfig, gameModel) =>
        val modelOutputDir = new Path(outputDir, s"all/$modelIdx").toString
        Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)
        val modelSpecDir = new Path(modelOutputDir, "model-spec").toString
        IOUtils.writeStringsToHDFS(Iterator(modelConfig), modelSpecDir, hadoopConfiguration, forceOverwrite = false)
        ModelProcessingUtils.saveGameModelsToHDFS(gameModel, featureShardIdToFeatureMapLoader, modelOutputDir,
          numberOfOutputFilesForRandomEffectModel, sparkContext)
        modelIdx += 1
      }
    }
  }

  /**
   * Run the driver
   */
  def run(): Unit = {
    val timer = new Timer

    // Process the output directory upfront and potentially fail the job early
    IOUtils.processOutputDir(outputDir, deleteOutputDirIfExists, sparkContext.hadoopConfiguration)

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
    val trainingLossFunctionEvaluator = prepareTrainingLossFunctionEvaluator(gameDataSet)
    timer.stop()
    logger.info(s"Time elapsed after training evaluator preparation: ${timer.durationSeconds} (s)\n")

    // Get rid of the largest object, which is no longer needed in the following code
    gameDataSet.unpersist()

    val validatingDataAndEvaluatorsOption = validateDirsOpt match {
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

    trainingDataSet.foreach { case (_, rddLike: RDDLike) => rddLike.unpersistRDD() }

    if (modelOutputMode != ModelOutputMode.NONE) {
      timer.start()
      val validatingDataAndEvaluatorOption = validatingDataAndEvaluatorsOption
          .map { case (validationData, evaluators) => (validationData, evaluators.head) }
      saveModelToHDFS(featureShardIdToFeatureMapMap, validatingDataAndEvaluatorOption, gameModelsMap)
      timer.stop()
      logger.info(s"Time elapsed after saving game models to HDFS: ${timer.durationSeconds} (s)\n")
    }
  }
}

object Driver {
  val DEFAULT_TREE_AGGREGATE_DEPTH = 1
  val DEEP_TREE_AGGREGATE_DEPTH = 2
  val TRACK_STATE = false
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
