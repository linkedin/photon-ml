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
package com.linkedin.photon.ml.cli.game.scoring

import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.cli.game.GAMEDriver
import com.linkedin.photon.ml.avro.AvroUtils
import com.linkedin.photon.ml.avro.data.{DataProcessingUtils, ScoreProcessingUtils}
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.{GameDatum, KeyValueScore}
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.util._
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.Map

/**
  * Driver for GAME full model scoring
  */
class Driver(val params: Params, val sparkContext: SparkContext, val logger: PhotonLogger)
    extends GAMEDriver(params, sparkContext, logger) {

  import params._

  protected val parallelism: Int = sparkContext.getConf.get("spark.default.parallelism",
    s"${sparkContext.getExecutorStorageStatus.length * 3}").toInt

  /**
   * Builds a GAME data set according to input data configuration
   *
   * @param featureShardIdToFeatureMapLoader A map of shard id to feature map loader
   * @return The prepared GAME data set
   */
  protected def prepareGameDataSet(featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader])
  : RDD[(Long, GameDatum)] = {

    val recordsPath = (dateRangeOpt, dateRangeDaysAgoOpt) match {
      // Specified as date range
      case (Some(trainDateRange), None) =>
        val dateRange = DateRange.fromDates(trainDateRange)
        IOUtils.getInputPathsWithinDateRange(inputDirs, dateRange, hadoopConfiguration, errorOnMissing = false)

      // Specified as a range of start days ago - end days ago
      case (None, Some(trainDateRangeDaysAgo)) =>
        val dateRange = DateRange.fromDaysAgo(trainDateRangeDaysAgo)
        IOUtils.getInputPathsWithinDateRange(inputDirs, dateRange, hadoopConfiguration, errorOnMissing = false)

      // Both types specified: illegal
      case (Some(_), Some(_)) =>
        throw new IllegalArgumentException(
          "Both trainDateRangeOpt and trainDateRangeDaysAgoOpt given. You must specify date ranges using only one " +
              "format.")

      // No range specified, just use the train dir
      case (None, None) => inputDirs.toSeq
    }
    logger.debug(s"Input records paths:\n${recordsPath.mkString("\n")}")
    val records = AvroUtils.readAvroFiles(sparkContext, recordsPath, parallelism)
    val recordsWithUniqueId = records.zipWithUniqueId().map(_.swap)
    val globalDataPartitioner = new LongHashPartitioner(records.partitions.length)

    val gameDataSet = DataProcessingUtils.getGameDataSetFromGenericRecords(
      recordsWithUniqueId,
      featureShardIdToFeatureSectionKeysMap,
      featureShardIdToFeatureMapLoader,
      randomEffectIdTypeSet,
      isResponseRequired = false)
      .partitionBy(globalDataPartitioner)
      .setName("Game data set with UIDs for scoring")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    logGameDataSet(gameDataSet)
    gameDataSet
  }

  /**
   * Log some statistics of the GAME data set for debugging purpose
   *
   * @param gameDataSet The GAME data set
   */
  private def logGameDataSet(gameDataSet: RDD[(Long, GameDatum)]): Unit = {
    // Log some simple summary info on the Game data set
    logger.debug(s"Summary for the GAME data set")
    val numSamples = gameDataSet.count()
    logger.debug(s"numSamples: $numSamples")
    randomEffectIdTypeSet.foreach { idType =>
      val numSamplesStats = gameDataSet.map { case (_, gameData) =>
        val idValue = gameData.idTypeToValueMap(idType)
        (idValue, 1)
      }
        .reduceByKey(_ + _)
        .values
        .stats()
      logger.debug(s"numSamples for $idType: $numSamplesStats")
    }
  }

  /**
   * Load the GAME model and score the GAME data set.
   *
   * @param featureShardIdToIndexMapLoader A map of feature shard id to feature map loader
   * @param gameDataSet The game data set
   * @return The scores
   */
  protected def scoreGameDataSet(
      featureShardIdToIndexMapLoader: Map[String, IndexMapLoader],
      gameDataSet: RDD[(Long, GameDatum)]): KeyValueScore = {

    // TODO: make the number of files written to HDFS to be configurable

    // Load the model from HDFS
    val gameModel = ModelProcessingUtils.loadGameModelFromHDFS(
      featureShardIdToIndexMapLoader, gameModelInputDir, sparkContext)

    logger.debug(s"Loaded game model summary:\n${gameModel.toSummaryString}")

    gameModel.score(gameDataSet)
  }

  /**
   * Save the computed scores to HDFS with auxiliary info
   *
   * @param gameDataSet The GAME data set
   * @param scores The computed scores
   */
  protected def saveScoresToHDFS(
      gameDataSet: RDD[(Long, GameDatum)],
      scores: KeyValueScore): Unit = {

    // Take the offset information into account when writing the scores to HDFS
    val scoredItems = gameDataSet.join(scores.scores).map { case (_, (gameDatum, score)) =>
      ScoredItem(score + gameDatum.offset, Some(gameDatum.response), gameDatum.weightOpt, gameDatum.idTypeToValueMap)
    }
    scoredItems.setName("Scored items").persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val numScoredItems = scoredItems.count()
    logger.info(s"Number of scored items to be written to HDFS: $numScoredItems (s)\n")
    val scoredItemsToBeSaved =
      if (numOutputFilesForScores > 0 && numOutputFilesForScores != scoredItems.partitions.length) {
        scoredItems.repartition(numOutputFilesForScores)
      } else {
        scoredItems
      }
    val scoresDir = Driver.getScoresDir(outputDir)
    ScoreProcessingUtils.saveScoredItemsToHDFS(scoredItemsToBeSaved, modelId = gameModelId, scoresDir)
    scoredItems.unpersist()
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
    val scores = scoreGameDataSet(featureShardIdToFeatureMapMap, gameDataSet)
    scores.persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    timer.stop()
    logger.info(s"Time elapsed after computing scores: ${timer.durationSeconds} (s)\n")

    timer.start()
    saveScoresToHDFS(gameDataSet, scores)
    timer.stop()
    logger.info(s"Time elapsed saving scores to HDFS: ${timer.durationSeconds} (s)\n")

    timer.start()
    evaluatorTypes.foreach { evaluatorType =>
      val evaluationMetricValue = Driver.evaluateScores(evaluatorType, scores, gameDataSet)
      logger.info(s"Evaluation metric value on scores with $evaluatorType: $evaluationMetricValue")
    }
    timer.stop()
    logger.info(s"Time elapsed after evaluating scores: ${timer.durationSeconds} (s)\n")
  }
}

object Driver {
  private val SCORES = "scores"
  private val LOGS = "logs"

  protected[scoring] def getScoresDir(outputDir: String): String = {
    new Path(outputDir, Driver.SCORES).toString
  }

  protected[scoring] def getLogsPath(outputDir: String): String = {
    new Path(outputDir, LOGS).toString
  }

  /**
   * Evaluate the computed scores with the given evaluator type
   * @param evaluatorType The evaluator type
   * @param scores The computed scores
   * @param gameDataSet The GAME data set
   */
  protected[scoring] def evaluateScores(
      evaluatorType: EvaluatorType,
      scores: KeyValueScore,
      gameDataSet: RDD[(Long, GameDatum)]): Double = {

    // Make sure the GAME data set makes sense
    val numSamplesWithNaNResponse = gameDataSet.filter(_._2.response.isNaN).count()
    require(numSamplesWithNaNResponse == 0,
      s"Number of data points with NaN found as response: $numSamplesWithNaNResponse. Make sure the responses are " +
        s"well defined in your data point in order to evaluate the computed scores with the specified " +
        s"evaluator $evaluatorType!")

    val evaluator = Evaluator.buildEvaluator(evaluatorType, gameDataSet)
    evaluator.evaluate(scores.scores)
  }

  /**
   * Main entry point
   */
  def main(args: Array[String]): Unit = {

    val timer = Timer.start()

    val params = Params.parseFromCommandLine(args)
    import params._

    val sc = SparkContextConfiguration.asYarnClient(applicationName, useKryo = true)

    val logsDir = getLogsPath(outputDir)
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
