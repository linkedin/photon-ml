/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.cli.game.scoring

import scala.collection.Map

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.cli.game.GameDriver
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.avro._
import com.linkedin.photon.ml.data.scoring.ModelDataScores
import com.linkedin.photon.ml.data.{GameConverters, GameDatum}
import com.linkedin.photon.ml.evaluation.{EvaluatorFactory, EvaluatorType, MultiEvaluatorType}
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.model.RandomEffectModel
import com.linkedin.photon.ml.util._

/**
 * Driver for GAME full model scoring.
 */
object GameScoringDriver extends GameDriver {

  //
  // Members
  //

  protected[scoring] var sc: SparkContext = _
  protected[scoring] var parameters: GameScoringParams = _
  protected[scoring] implicit var logger: PhotonLogger = _

  val SCORES_DIR = "scores"

  //
  // Params trait extensions
  //

  /**
   * Dummy function until refactored.
   *
   * @param extra Additional parameters which should overwrite the values being copied
   * @return This object
   */
  override def copy(extra: ParamMap): Params = this

  //
  // Scoring driver functions
  //

  /**
   * Builds a GAME data set according to input data configuration.
   *
   * @param featureShardIdToFeatureMapLoader A map of shard id to feature map loader
   * @return The prepared GAME data set
   */
  protected def prepareGameDataSet(
    featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader]): RDD[(Long, GameDatum)] = {

    // Handle date range input
    val dateRangeOpt = IOUtils.resolveRange(
      parameters.dateRangeOpt.map(DateRange.fromDateString),
      parameters.dateRangeDaysAgoOpt.map(DaysRange.fromDaysString))
    val recordsPath = pathsForDateRange(parameters.inputDirs.toSet[String].map(new Path(_)), dateRangeOpt)

    logger.debug(s"Input records paths:\n${recordsPath.mkString("\n")}")

    val featureShardIdToFeatureSectionKeysMap = parameters.featureShardIdToFeatureSectionKeysMap
    val parallelism = sc.getConf.get("spark.default.parallelism", s"${sc.getExecutorStorageStatus.length * 3}").toInt
    val dataReader = new AvroDataReader(sc)
    val data = dataReader.readMerged(
      recordsPath.map(_.toString),
      featureShardIdToFeatureMapLoader.toMap,
      parameters.featureShardIdToFeatureSectionKeysMap,
      parallelism)
    val partitioner = new LongHashPartitioner(parallelism)
    val idTagSet: Set[String] = parameters.randomEffectTypeSet ++
      parameters.evaluatorTypes.map(MultiEvaluatorType.getMultiEvaluatorIdTags).getOrElse(Seq())
    val gameDataSet = GameConverters
      .getGameDataSetFromDataFrame(
        data,
        featureShardIdToFeatureSectionKeysMap.keys.toSet,
        idTagSet,
        isResponseRequired = false,
        parameters.inputColumnsNames)
      .partitionBy(partitioner)
      .setName("Game data set with UIDs for scoring")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    if (parameters.logDatasetAndModelStats) {
      logGameDataSet(gameDataSet)
    }

    gameDataSet
  }

  /**
   * Log some statistics of the GAME data set for debugging purpose.
   *
   * @param gameDataSet The GAME data set
   */
  private def logGameDataSet(gameDataSet: RDD[(Long, GameDatum)]): Unit = {
    // Log some simple summary info on the GAME data set
    logger.debug(s"Summary for the GAME data set")
    val numSamples = gameDataSet.count()
    logger.debug(s"numSamples: $numSamples")
    parameters.randomEffectTypeSet.foreach { idTag =>
      val numSamplesStats = gameDataSet.map { case (_, gameData) =>
          val idValue = gameData.idTagToValueMap(idTag)
          (idValue, 1)
        }
        .reduceByKey(_ + _)
        .values
        .stats()
      logger.debug(s"numSamples for $idTag: $numSamplesStats")
    }
  }

  /**
   * Load the GAME model and score the GAME data set.
   *
   * @param featureShardIdToIndexMapLoader A map of feature shard id to feature map loader
   * @param gameDataSet The GAME data set
   * @return The scores
   */
  protected def scoreGameDataSet(
      featureShardIdToIndexMapLoader: Map[String, IndexMapLoader],
      gameDataSet: RDD[(Long, GameDatum)]): ModelDataScores = {

    // TODO: make the number of files written to HDFS configurable

    // Load the model from HDFS, ignoring the feature index loader
    val (gameModel, _) = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      new Path(parameters.gameModelInputDir),
      StorageLevel.VERY_FREQUENT_REUSE_RDD_STORAGE_LEVEL,
      Some(featureShardIdToIndexMapLoader))

    if (parameters.logDatasetAndModelStats) {
      logger.debug(s"Loaded game model summary:\n${gameModel.toSummaryString}")
    }

    // Need to split these calls to keep correct return type
    val scores = gameModel.score(gameDataSet)
    val storageLevel = if (parameters.spillScores) {
      StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL
    } else {
      StorageLevel.VERY_FREQUENT_REUSE_RDD_STORAGE_LEVEL
    }

    scores.persistRDD(storageLevel).materialize()

    gameDataSet.unpersist()
    gameModel.toMap.foreach {
      case (_, model: RandomEffectModel) => model.unpersistRDD()
      case _ =>
    }

    scores
  }

  /**
   * Save the computed scores to HDFS with auxiliary info.
   *
   * @param scores The computed scores
   */
  protected def saveScoresToHDFS(scores: ModelDataScores): Unit = {

    // Take the offset information into account when writing the scores to HDFS
    val scoredItems = scores.scores.map { case (_, scoredGameDatum) =>
      ScoredItem(
        scoredGameDatum.score + scoredGameDatum.offset,
        Some(scoredGameDatum.response),
        Some(scoredGameDatum.weight),
        scoredGameDatum.idTagToValueMap)
    }

    if (parameters.logDatasetAndModelStats) {
      // Persist scored items here since we introduce multiple passes
      scoredItems.setName("Scored items").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)

      val numScoredItems = scoredItems.count()
      logger.info(s"Number of scored items to be written to HDFS: $numScoredItems \n")
    }

    val scoredItemsToBeSaved =
      if (parameters.numOutputFilesForScores > 0 && parameters.numOutputFilesForScores != scoredItems.partitions.length) {
        scoredItems.repartition(parameters.numOutputFilesForScores)
      } else {
        scoredItems
      }
    val scoresDir = new Path(parameters.outputDir, SCORES_DIR)

    ScoreProcessingUtils.saveScoredItemsToHDFS(scoredItemsToBeSaved, modelId = parameters.gameModelId, scoresDir.toString)
    scoredItems.unpersist()
  }

  /**
   * Run the driver.
   */
  protected[scoring] def run(): Unit = {
    val timer = new Timer

    // Process the output directory upfront and potentially fail the job early
    IOUtils.processOutputDir(new Path(parameters.outputDir), parameters.deleteOutputDirIfExists, sc.hadoopConfiguration)

    timer.start()
    // TODO: Remove after updated CLI
    val featureShardConfigs = parameters.featureShardIdToFeatureSectionKeysMap.map { case (featureShardId, featureBags) =>
      (featureShardId, FeatureShardConfiguration(featureBags, hasIntercept = true))
    }
    set(featureShardConfigurations, featureShardConfigs)
    set(featureBagsDirectory, new Path(parameters.featureNameAndTermSetInputPath))
    parameters.offHeapIndexMapDir.foreach { dir =>
      set(offHeapIndexMapDirectory, new Path(dir))
      set(offHeapIndexMapPartitions, parameters.offHeapIndexMapNumPartitions)
    }
    val featureShardIdToFeatureMapMap = prepareFeatureMaps()
    timer.stop()
    logger.info(s"Time elapsed after preparing feature maps: ${timer.durationSeconds} (s)\n")

    timer.start()
    val gameDataSet = prepareGameDataSet(featureShardIdToFeatureMapMap)
    timer.stop()
    logger.info(s"Time elapsed after game data set preparation: ${timer.durationSeconds} (s)\n")

    timer.start()
    val scores = scoreGameDataSet(featureShardIdToFeatureMapMap, gameDataSet)
    timer.stop()
    logger.info(s"Time elapsed after computing scores: ${timer.durationSeconds} (s)\n")

    timer.start()
    saveScoresToHDFS(scores)
    timer.stop()
    logger.info(s"Time elapsed saving scores to HDFS: ${timer.durationSeconds} (s)\n")

    timer.start()
    parameters.evaluatorTypes.foreach(_.foreach { evaluatorType =>
      val evaluationMetricValue = evaluateScores(evaluatorType, scores, gameDataSet)
      logger.info(s"Evaluation metric value on scores with $evaluatorType: $evaluationMetricValue")
    })
    timer.stop()
    logger.info(s"Time elapsed after evaluating scores: ${timer.durationSeconds} (s)\n")
  }

  /**
   * Evaluate the computed scores with the given evaluator type
   *
   * @param evaluatorType The evaluator type
   * @param scores The computed scores
   * @param gameDataSet The GAME data set
   * @return The evaluation metric
   */
  protected[scoring] def evaluateScores(
      evaluatorType: EvaluatorType,
      scores: ModelDataScores,
      gameDataSet: RDD[(Long, GameDatum)]): Double = {

    // Make sure the GAME data set makes sense
    val numSamplesWithNaNResponse = gameDataSet.filter(_._2.response.isNaN).count()
    require(numSamplesWithNaNResponse == 0,
      s"Number of data points with NaN found as response: $numSamplesWithNaNResponse. Make sure the responses are " +
        s"well defined in your data point in order to evaluate the computed scores with the specified " +
        s"evaluator $evaluatorType")

    val evaluator = EvaluatorFactory.buildEvaluator(evaluatorType, gameDataSet)
    evaluator.evaluate(scores.scores.mapValues(_.score))
  }

  /**
   * Entry point to the driver.
   *
   * @param args The command line arguments for the job
   */
  def main(args: Array[String]): Unit = {

    val timer = Timer.start()

    parameters = GameScoringParams.parseFromCommandLine(args)
    sc = SparkContextConfiguration.asYarnClient(parameters.applicationName, useKryo = true)
    logger = new PhotonLogger(new Path(parameters.outputDir), sc)
    // TODO: This Photon log level should be made configurable
    logger.setLogLevel(PhotonLogger.LogLevelDebug)

    try {
      logger.debug(params.toString + "\n")

      run()

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
