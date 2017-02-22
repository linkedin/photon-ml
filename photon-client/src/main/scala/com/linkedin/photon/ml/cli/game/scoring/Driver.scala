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

import scala.collection.Map

import org.apache.hadoop.fs.Path
import org.slf4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.cli.game.GAMEDriver
import com.linkedin.photon.ml.avro.data.ScoreProcessingUtils
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.{AvroDataReader, GameConverters, GameDatum, KeyValueScore}
import com.linkedin.photon.ml.evaluation.{EvaluatorFactory, EvaluatorType}
import com.linkedin.photon.ml.util._

/**
 * Driver for GAME full model scoring.
 */
class Driver(val params: Params, val sparkContext: SparkContext, val logger: Logger)
  extends GAMEDriver(params, sparkContext, logger) {

  import params._

  protected[game] val idTypeSet: Set[String] = {
    randomEffectTypeSet ++ getShardedEvaluatorIdTypes
  }

  /**
   * Builds a GAME data set according to input data configuration.
   *
   * @param featureShardIdToFeatureMapLoader A map of shard id to feature map loader
   * @return The prepared GAME data set
   */
  protected def prepareGameDataSet(
    featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader]): RDD[(Long, GameDatum)] = {

    val recordsPath = pathsForDateRange(inputDirs, dateRangeOpt, dateRangeDaysAgoOpt)
    logger.debug(s"Input records paths:\n${recordsPath.mkString("\n")}")

    val gameDataPartitioner = new LongHashPartitioner(parallelism)

    val dataReader = new AvroDataReader(sparkContext)
    val data = dataReader.readMerged(
      recordsPath,
      featureShardIdToFeatureMapLoader.toMap,
      featureShardIdToFeatureSectionKeysMap,
      parallelism)

    val gameDataSet = GameConverters.getGameDataSetFromDataFrame(
      data,
      featureShardIdToFeatureSectionKeysMap.keys.toSet,
      idTypeSet,
      isResponseRequired = false)
      .partitionBy(gameDataPartitioner)
      .setName("Game data set with UIDs for scoring")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    logGameDataSet(gameDataSet)
    gameDataSet
  }

  /**
   * Log some statistics of the GAME data set for debugging purpose.
   *
   * @param gameDataSet The GAME data set
   */
  private def logGameDataSet(gameDataSet: RDD[(Long, GameDatum)]): Unit = {
    // Log some simple summary info on the Game data set
    logger.debug(s"Summary for the GAME data set")
    val numSamples = gameDataSet.count()
    logger.debug(s"numSamples: $numSamples")
    randomEffectTypeSet.foreach { idType =>
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

    // TODO: make the number of files written to HDFS configurable

    // Load the model from HDFS, ignoring the feature index loader
    val (gameModel, _) =
      ModelProcessingUtils.loadGameModelFromHDFS(Some(featureShardIdToIndexMapLoader), gameModelInputDir, sparkContext)

    logger.debug(s"Loaded game model summary:\n${gameModel.toSummaryString}")

    val scores = gameModel.score(gameDataSet).persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL).materialize()

    gameModel.unpersist()
    scores
  }

  /**
   * Save the computed scores to HDFS with auxiliary info.
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
   * Run the driver.
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

  /**
   *
   * @param outputDir
   * @return
   */
  protected[scoring] def getScoresDir(outputDir: String): String = {
    new Path(outputDir, Driver.SCORES).toString
  }

  /**
   *
   * @param outputDir
   * @return
   */
  protected[scoring] def getLogsPath(outputDir: String): String = {
    new Path(outputDir, LOGS).toString
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
      scores: KeyValueScore,
      gameDataSet: RDD[(Long, GameDatum)]): Double = {

    // Make sure the GAME data set makes sense
    val numSamplesWithNaNResponse = gameDataSet.filter(_._2.response.isNaN).count()
    require(numSamplesWithNaNResponse == 0,
      s"Number of data points with NaN found as response: $numSamplesWithNaNResponse. Make sure the responses are " +
        s"well defined in your data point in order to evaluate the computed scores with the specified " +
        s"evaluator $evaluatorType!")

    val evaluator = EvaluatorFactory.buildEvaluator(evaluatorType, gameDataSet)
    evaluator.evaluate(scores.scores)
  }

  /**
   * Main entry point.
   *
   * @param args
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
