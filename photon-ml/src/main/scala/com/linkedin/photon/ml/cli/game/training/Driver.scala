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


import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.avro.AvroUtils
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data._
import org.apache.spark.rdd.RDD

import scala.collection.Map

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext

import com.linkedin.photon.ml.algorithm._
import com.linkedin.photon.ml.avro.data.{NameAndTerm, DataProcessingUtils, NameAndTermFeatureSetContainer}
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.model.Model
import com.linkedin.photon.ml.optimization.game.{
  FactoredRandomEffectOptimizationProblem, RandomEffectOptimizationProblem, OptimizationProblem}
import com.linkedin.photon.ml.projector.IdentityProjection
import com.linkedin.photon.ml.supervised.TaskType._
import com.linkedin.photon.ml.util._


/**
 * Driver for GAME full model training
 *
 * @author xazhang
 */
class Driver(val params: Params, val sparkContext: SparkContext, val logger: PhotonLogger) {

  import params._

  protected val hadoopConfiguration = sparkContext.hadoopConfiguration

  protected val isAddingIntercept = true

  /**
   * Builds feature name-and-term to index maps according to configuration
   *
   * @return a map of shard id to feature map
   */
  def prepareFeatureMaps(): Map[String, Map[NameAndTerm, Int]] = {
    val allFeatureSectionKeys = featureShardIdToFeatureSectionKeysMap.values.reduce(_ ++ _)
    val nameAndTermFeatureSetContainer = NameAndTermFeatureSetContainer.loadFromTextFiles(
      featureNameAndTermSetInputPath, allFeatureSectionKeys, hadoopConfiguration)

    val featureShardIdToFeatureMapMap =
      featureShardIdToFeatureSectionKeysMap.map { case (shardId, featureSectionKeys) =>
        val featureMap = nameAndTermFeatureSetContainer.getFeatureNameAndTermToIndexMap(featureSectionKeys,
          isAddingIntercept)
        (shardId, featureMap)
      }
    featureShardIdToFeatureMapMap.foreach { case (shardId, featureMap) =>
      logger.logDebug(s"Feature shard ID: $shardId, number of features: ${featureMap.size}")
    }
    featureShardIdToFeatureMapMap
  }

  /**
   * Builds a GAME dataset according to input data configuration
   *
   * @param featureShardIdToFeatureMapMap a map of shard id to feature map
   * @return the prepared GAME dataset
   */
  def prepareGameDataSet(featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]]): RDD[(Long, GameData)] = {

    val trainingRecordsPath = trainDateRangeOpt match {
      case Some(trainDateRange) =>
        val Array(startDate, endDate) = trainDateRange.split("-")
        IOUtils.getInputPathsWithinDateRange(trainDirs, startDate, endDate, hadoopConfiguration, errorOnMissing = false)
      case None => trainDirs.toSeq
    }
    logger.logDebug(s"Training records paths:\n${trainingRecordsPath.mkString("\n")}")

    // Determine the number of fixed effect partitions. Default to 0 if we have no fixed effects.
    val numFixedEffectPartitions = if (fixedEffectDataConfigurations.nonEmpty) {
      fixedEffectDataConfigurations.values.map(_.numPartitions).max
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
    val globalDataPartitioner = new LongHashPartitioner(records.partitions.length)

    val randomEffectIdSet = randomEffectDataConfigurations.values.map(_.randomEffectId).toSet
    val gameDataSet = DataProcessingUtils.parseAndGenerateGameDataSetFromGenericRecords(records,
      featureShardIdToFeatureSectionKeysMap, featureShardIdToFeatureMapMap, randomEffectIdSet)
        .partitionBy(globalDataPartitioner)
        .setName("Training Game data set")
        .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    gameDataSet.count()
    gameDataSet
  }

  /**
   * Prepares the training dataset
   *
   * @param gameDataSet the input dataset
   * @return the training dataset
   */
  def prepareTrainingDataSet(gameDataSet: RDD[(Long, GameData)]): Map[String, DataSet[_ <: DataSet[_]]] = {

    val fixedEffectDataSets = fixedEffectDataConfigurations.map { case (id, fixedEffectDataConfiguration) =>
      val fixedEffectDataSet = FixedEffectDataSet.buildWithConfiguration(gameDataSet, fixedEffectDataConfiguration)
        .setName(s"Fixed effect data set with id $id")
        .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
      logger.logDebug(s"Fixed effect data set with id $id summary:\n${fixedEffectDataSet.toSummaryString}\n")
      (id, fixedEffectDataSet)
    }

    // Prepare the per-random effect partitioner
    val randomEffectPartitionerMap = randomEffectDataConfigurations.map { case (id, randomEffectDataConfiguration) =>
      val numPartitions = randomEffectDataConfiguration.numPartitions
      val randomEffectId = randomEffectDataConfiguration.randomEffectId
      (id, RandomEffectIdPartitioner.generateRandomEffectIdPartitionerFromGameDataSet(numPartitions,
        randomEffectId, gameDataSet))
    }.toMap

    // Prepare the random effect data sets
    val randomEffectDataSet = randomEffectDataConfigurations.map { case (id, randomEffectDataConfiguration) =>
      val randomEffectPartitioner = randomEffectPartitionerMap(id)
      val rawRandomEffectDataSet = RandomEffectDataSet.buildWithConfiguration(gameDataSet,
        randomEffectDataConfiguration, randomEffectPartitioner)
          .setName(s"Random effect data set with coordinate id $id")
          .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
          .materialize()
      val projectorType = randomEffectDataConfiguration.projectorType
      val randomEffectDataSet = projectorType match {
        case IdentityProjection => rawRandomEffectDataSet
        case _ =>
          val randomEffectDataSetInProjectedSpace =
            RandomEffectDataSetInProjectedSpace.buildWithProjectorType(rawRandomEffectDataSet, projectorType)
                .setName(s"Random effect data set in projected space with coordinate id $id")
                .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
                .materialize()
          // Only un-persist the active data and passive data, because randomEffectDataSet and
          // randomEffectDataSetInProjectedSpace share globalIdToIndividualIds and other RDDs/Broadcasts
          rawRandomEffectDataSet.activeData.unpersist()
          rawRandomEffectDataSet.passiveDataOption.foreach(_.unpersist())
          randomEffectDataSetInProjectedSpace
      }
      logger.logDebug(s"Random effect data set with id $id summary:\n${randomEffectDataSet.toSummaryString}\n")
      (id, randomEffectDataSet)
    }

    fixedEffectDataSets ++ randomEffectDataSet
  }

  /**
   * Creates the training evaluator
   *
   * @param gameDataSet the input dataset
   * @return the training evaluator
   */
  def prepareTrainingEvaluator(gameDataSet: RDD[(Long, GameData)]): Evaluator = {
    val labelAndOffsetAndWeights = gameDataSet.mapValues(gameData =>
      (gameData.response, gameData.offset, gameData.weight)
    ).setName("Training label and offset and weights").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    labelAndOffsetAndWeights.count()
    taskType match {
      case LOGISTIC_REGRESSION =>
        new LogisticLossEvaluator(labelAndOffsetAndWeights)
      case LINEAR_REGRESSION =>
        new SquaredLossEvaluator(labelAndOffsetAndWeights)
      case _ =>
        throw new UnsupportedOperationException(s"Task type: $taskType is not supported to create training evaluator")
    }
  }

  /**
   * Creates the validation evaluator
   *
   * @param gameDataSet the input dataset
   * @return the validation evaluator
   */
  def prepareValidatingEvaluator(
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]]): Option[(RDD[(Long, GameData)], Evaluator)] = {

    validateDirsOpt match {
      case Some(validatingDirs) =>
        // Read and parse the validating activities
        val validatingRecordsPath = validateDateRangeOpt match {
          case Some(validateDateRange) =>
            val Array(startDate, endDate) = validateDateRange.split("-")
            IOUtils.getInputPathsWithinDateRange(validatingDirs, startDate, endDate, sparkContext.hadoopConfiguration,
              errorOnMissing = false)
          case None => validatingDirs.toSeq
        }
        logger.logDebug(s"Validating records paths:\n${validatingRecordsPath.mkString("\n")}")
        val records = AvroUtils.readAvroFiles(sparkContext, validatingRecordsPath, minPartitionsForValidation)
        val partitioner = new LongHashPartitioner(records.partitions.length)
        // filter out features that validating data are included in the black list

        val randomEffectIdSet = randomEffectDataConfigurations.values.map(_.randomEffectId).toSet
        val gameDataSet = DataProcessingUtils.parseAndGenerateGameDataSetFromGenericRecords(records,
          featureShardIdToFeatureSectionKeysMap, featureShardIdToFeatureMapMap, randomEffectIdSet)
            .partitionBy(partitioner)
            .setName("Validating Game data set")
            .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

        // Log some simple summary info on the Game data set
        logger.logDebug(s"Summary for the validating Game data set")
        val numSamples = gameDataSet.count()
        logger.logDebug(s"numSamples: $numSamples")
        val responseSum = gameDataSet.values.map(_.response).sum()
        logger.logDebug(s"responseSum: $responseSum")
        val weightSum = gameDataSet.values.map(_.weight).sum()
        logger.logDebug(s"weightSum: $weightSum")
        val randomEffectIdToIndividualIdMap = gameDataSet.values.first().randomEffectIdToIndividualIdMap
        randomEffectIdToIndividualIdMap.keySet.foreach { randomEffectId =>
          val dataStats = gameDataSet.values.map { gameData =>
            val individualId = gameData.randomEffectIdToIndividualIdMap(randomEffectId)
            (individualId, (gameData.response, 1))
          }.reduceByKey { case ((responseSum1, numSample1), (responseSum2, numSample2)) =>
            (responseSum1 + responseSum2, numSample1 + numSample2)
          }.cache()
          val responseSumStats = dataStats.values.map(_._1).stats()
          val numSamplesStats = dataStats.values.map(_._2).stats()
          logger.logDebug(s"numSamplesStats for $randomEffectId: $numSamplesStats")
          logger.logDebug(s"responseSumStats for $randomEffectId: $responseSumStats")
        }

        val validatingLabelAndOffsets = gameDataSet.mapValues(gameData => (gameData.response, gameData.offset))
            .setName(s"Validating labels and offsets").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
        validatingLabelAndOffsets.count()

        val evaluator =
          taskType match {
            case LOGISTIC_REGRESSION =>
              new BinaryClassificationEvaluator(validatingLabelAndOffsets)
            case LINEAR_REGRESSION =>
              val validatingLabelAndOffsetAndWeights = validatingLabelAndOffsets.mapValues { case (label, offset) =>
                (label, offset, 1.0)
              }
              new RMSEEvaluator(validatingLabelAndOffsetAndWeights)
            case _ =>
              throw new UnsupportedOperationException(s"Task type: $taskType is not supported to create validating " +
                  s"evaluator")
          }
        val randomScores = gameDataSet.mapValues(_ => math.random)
        val metric = evaluator.evaluate(randomScores)
        logger.logInfo(s"Random guessing based baseline evaluation metric: $metric")
        Some((gameDataSet, evaluator))
      case None => None
    }
  }

  /**
   * Train GAME models. This method builds a coordinate descent optimization problem from the individual optimization
   * problems for the fixed effect, random effect, and factored random effect models.
   *
   * @param dataSets the training datasets
   * @param trainingEvaluator the training evaluator
   * @param validatingDataAndEvaluatorOption optional validation dataset and evaluator
   * @return trained GAME models
   */
  def train(
      dataSets: Map[String, DataSet[_ <: DataSet[_]]],
      trainingEvaluator: Evaluator,
      validatingDataAndEvaluatorOption: Option[(RDD[(Long, GameData)], Evaluator)]): Map[String, Map[String, Model]] = {

    val gameModels = for (
        fixedEffectOptimizationConfiguration <- fixedEffectOptimizationConfigurations;
        randomEffectOptimizationConfiguration <- randomEffectOptimizationConfigurations;
        factoredRandomEffectOptimizationConfiguration <- factoredRandomEffectOptimizationConfigurations) yield {

      val modelConfig = fixedEffectOptimizationConfiguration.mkString("\n") + "\n" +
          randomEffectOptimizationConfiguration.mkString("\n") + "\n" +
          factoredRandomEffectOptimizationConfiguration.mkString("\n")
      val startTime = System.nanoTime()
      logger.logInfo(s"Start to train the game model with the following config:\n$modelConfig\n")

      // For each model, create optimization coordinates for the fixed effect, random effect, and factored random effect
      // models
      val coordinates = updatingSequence.map { coordinateId =>
        val coordinate = dataSets(coordinateId) match {
          case fixedEffectDataSet: FixedEffectDataSet =>
            // Fixed effect coordinate
            val optimizationConfiguration = fixedEffectOptimizationConfiguration(coordinateId)

            // If number of features is from moderate to large (>200000), then use tree aggregate,
            // otherwise use aggregate.
            val treeAggregateDepth = if (fixedEffectDataSet.numFeatures < 200000) 1 else 2
            val optimizationProblem =
              OptimizationProblem.buildOptimizationProblem(taskType, optimizationConfiguration)
            optimizationProblem.lossFunction.treeAggregateDepth = treeAggregateDepth
            println(s"Set treeAggregateDepth to ${optimizationProblem.objectiveFunction.treeAggregateDepth}")
            new FixedEffectCoordinate(fixedEffectDataSet, optimizationProblem)

          case randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace =>
            // Random effect coordinate
            val optimizationConfiguration = randomEffectOptimizationConfiguration(coordinateId)
            val randomEffectOptimizationProblem =
              RandomEffectOptimizationProblem.buildRandomEffectOptimizationProblem(taskType,
                optimizationConfiguration, randomEffectDataSetInProjectedSpace)
                  .setName(s"Random effect optimization problem of coordinate $coordinateId")
                  .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
            new RandomEffectCoordinateInProjectedSpace(randomEffectDataSetInProjectedSpace,
              randomEffectOptimizationProblem)

          case randomEffectDataSet: RandomEffectDataSet =>
            // Factored random effect coordinate
            val (randomEffectOptimizationConfiguration, latentFactorOptimizationConfiguration,
            mfOptimizationConfiguration) = factoredRandomEffectOptimizationConfiguration(coordinateId)
            val factoredRandomEffectOptimizationProblem =
               FactoredRandomEffectOptimizationProblem.buildFactoredRandomEffectOptimizationProblem(taskType,
                 randomEffectOptimizationConfiguration, latentFactorOptimizationConfiguration,
                 mfOptimizationConfiguration, randomEffectDataSet)
                   .setName(s"Factored random effect optimization problem of coordinate $coordinateId")
                   .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
            new FactoredRandomEffectCoordinate(randomEffectDataSet, factoredRandomEffectOptimizationProblem)

          case dataSet =>
            throw new UnsupportedOperationException(s"Data set of type ${dataSet.getClass} is not supported")
        }
        Pair[String, Coordinate[_ <: DataSet[_], _ <: Coordinate[_, _]]](coordinateId, coordinate)
      }
      val coordinateDescent = new CoordinateDescent(coordinates, trainingEvaluator, validatingDataAndEvaluatorOption,
        logger)
      val gameModel = coordinateDescent.run(numIterations)
      val timeElapsed = (System.nanoTime() - startTime) * 1e-9
      logger.logInfo(s"Finished training model with the following config:\n$modelConfig\n" +
          s"Time elapsed: $timeElapsed (s)\n")
      logger.flush()
      (modelConfig, gameModel)
    }

    gameModels.toMap
  }

  /**
   * Write the learned GAME model to HDFS
   *
   * @param featureShardIdToFeatureMapMap a map of shard id to feature map
   * @param validatingDataAndEvaluatorOption optional validation dataset and evaluator
   * @param gameModelsMap GAME models
   */
  def saveModelToHDFS(
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      validatingDataAndEvaluatorOption: Option[(RDD[(Long, GameData)], Evaluator)],
      gameModelsMap: Map[String, Map[String, Model]]) {

    val combinedGameModelsMap = gameModelsMap.map { case (modelName, gameModel) =>
      val collapsedGameModel = ModelProcessingUtils.collapseGameModel(gameModel, sparkContext).values
      collapsedGameModel.foreach {
        case rddLike: RDDLike => rddLike.persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
        case _ =>
      }
      (modelName, collapsedGameModel)
    }

    // Write the best model to HDFS
    validatingDataAndEvaluatorOption match {
      case Some((validatingData, evaluator)) =>
        val (bestModelConfig, evaluationResult) =
          combinedGameModelsMap
              .mapValues(_.map(_.score(validatingData))
              .reduce(_ + _).scores).mapValues(evaluator.evaluate)
              .reduce((result1, result2) => if (result1._2 > result2._2) result1 else result2)
        val bestGameModel = combinedGameModelsMap(bestModelConfig)
        logger.logInfo(s"The selected model has the following config:\n$bestModelConfig\nModel summary:" +
            s"\n${bestGameModel.map(_.toSummaryString).mkString("\n")}\n\nEvaluation result is : $evaluationResult")
        val modelOutputDir = new Path(outputDir, "best").toString
        Utils.deleteHDFSDir(modelOutputDir, sparkContext.hadoopConfiguration)
        Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)
        val modelSpecDir = new Path(modelOutputDir, "model-spec").toString
        IOUtils.writeStringsToHDFS(Iterator(bestModelConfig), modelSpecDir, hadoopConfiguration, forceOverwrite = false)
        ModelProcessingUtils.saveGameModelsToHDFS(bestGameModel, featureShardIdToFeatureMapMap, modelOutputDir,
          sparkContext)
      case _ =>
    }

    // Write all models to HDFS
    var modelIdx = 0
    combinedGameModelsMap.foreach { case (modelConfig, gameModel) =>
      val modelOutputDir = new Path(outputDir, s"all/$modelIdx").toString
      Utils.deleteHDFSDir(modelOutputDir, sparkContext.hadoopConfiguration)
      Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)
      val modelSpecDir = new Path(modelOutputDir, "model-spec").toString
      IOUtils.writeStringsToHDFS(Iterator(modelConfig), modelSpecDir, hadoopConfiguration, forceOverwrite = false)
      ModelProcessingUtils.saveGameModelsToHDFS(gameModel, featureShardIdToFeatureMapMap, modelOutputDir, sparkContext)
      modelIdx += 1
    }
  }

  /**
   * Run the driver
   */
  def run(): Unit = {

    var startTime = System.nanoTime()
    val featureShardIdToFeatureMapMap = prepareFeatureMaps()
    val initializationTime = (System.nanoTime() - startTime) * 1e-9
    logger.logInfo(s"Time elapsed after preparing feature maps: $initializationTime (s)\n")
    logger.flush()

    startTime = System.nanoTime()
    val gameDataSet = prepareGameDataSet(featureShardIdToFeatureMapMap)
    val gameDataPreparationTime = (System.nanoTime() - startTime) * 1e-9
    logger.logInfo(s"Time elapsed after game data set preparation: $gameDataPreparationTime (s)\n")
    logger.flush()

    startTime = System.nanoTime()
    val trainingDataSet = prepareTrainingDataSet(gameDataSet)
    val trainingDataSetPreparationTime = (System.nanoTime() - startTime) * 1e-9
    logger.logInfo(s"Time elapsed after training data set preparation: $trainingDataSetPreparationTime (s)\n")
    logger.flush()

    startTime = System.nanoTime()
    val trainingEvaluator = prepareTrainingEvaluator(gameDataSet)
    val trainingEvaluatorPreparationTime = (System.nanoTime() - startTime) * 1e-9
    logger.logInfo(s"Time elapsed after training evaluator preparation: $trainingEvaluatorPreparationTime (s)\n")
    logger.flush()

    // Get rid of the largest object, which is no longer needed in the following code
    gameDataSet.unpersist()

    startTime = System.nanoTime()
    val validatingDataAndEvaluatorOption = prepareValidatingEvaluator(featureShardIdToFeatureMapMap)
    if (validatingDataAndEvaluatorOption.isDefined) {
      val validatingEvaluatorPreparationTime = (System.nanoTime() - startTime) * 1e-9
      logger.logInfo(s"Time elapsed after validating evaluator preparation: $validatingEvaluatorPreparationTime (s)\n")
      logger.flush()
    }

    startTime = System.nanoTime()
    val gameModelsMap = train(trainingDataSet, trainingEvaluator, validatingDataAndEvaluatorOption)
    val trainingTime = (System.nanoTime() - startTime) * 1e-9
    logger.logInfo(s"Time elapsed after game model training: $trainingTime (s)\n")
    logger.flush()

    trainingDataSet.foreach { case (_, rddLike: RDDLike) => rddLike.unpersistRDD() }

    if (isSavingModelsToHDFS) {
      startTime = System.nanoTime()
      saveModelToHDFS(featureShardIdToFeatureMapMap, validatingDataAndEvaluatorOption, gameModelsMap)
      val savingModelTime = (System.nanoTime() - startTime) * 1e-9
      logger.logInfo(s"Time elapsed after saving game models to HDFS: $savingModelTime (s)\n")
      logger.flush()
    }
  }
}

object Driver {
  val LOGS = "logs"

  /**
   * Main entry point
   */
  def main(args: Array[String]): Unit = {

    val startTime = System.nanoTime()

    val params = Params.parseFromCommandLine(args)
    import params._

    val sc = SparkContextConfiguration.asYarnClient(applicationName)

    val logsDir = new Path(outputDir, LOGS).toString
    Utils.createHDFSDir(logsDir, sc.hadoopConfiguration)
    val logger = new PhotonLogger(logsDir, sc.hadoopConfiguration)
    logger.logDebug(params.toString + "\n")
    logger.flush()
    val job = new Driver(params, sc, logger)
    job.run()

    val timeElapsed = (System.nanoTime() - startTime) * 1e-9 / 60
    logger.logInfo(s"Overall time elapsed $timeElapsed minutes")
    logger.flush()

    logger.close()
    sc.stop()
  }
}
