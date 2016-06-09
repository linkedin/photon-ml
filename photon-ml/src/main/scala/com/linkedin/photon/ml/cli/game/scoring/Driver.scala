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
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.avro.AvroUtils
import com.linkedin.photon.ml.avro.data.{DataProcessingUtils, NameAndTerm, NameAndTermFeatureSetContainer}
import com.linkedin.photon.ml.avro.data.ScoreProcessingUtils
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.{GameDatum, KeyValueScore}
import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.util._


/**
 * Driver for GAME full model scoring
 */
class Driver(val params: Params, val sparkContext: SparkContext, val logger: PhotonLogger) {

  import params._

  protected val parallelism: Int = sparkContext.getConf.get("spark.default.parallelism",
    s"${sparkContext.getExecutorStorageStatus.length * 3}").toInt
  protected val hadoopConfiguration = sparkContext.hadoopConfiguration

  /**
   * Builds feature name-and-term to index maps according to configuration
   *
   * @return a map of shard id to feature map
   */
  protected def prepareFeatureMaps(): Map[String, Map[NameAndTerm, Int]] = {

    val allFeatureSectionKeys = featureShardIdToFeatureSectionKeysMap.values.reduce(_ ++ _)
    val nameAndTermFeatureSetContainer = NameAndTermFeatureSetContainer.readNameAndTermFeatureSetContainerFromTextFiles(
      featureNameAndTermSetInputPath, allFeatureSectionKeys, hadoopConfiguration)

    val featureShardIdToFeatureMapMap =
      featureShardIdToFeatureSectionKeysMap.map { case (shardId, featureSectionKeys) =>
        val featureMap = nameAndTermFeatureSetContainer.getFeatureNameAndTermToIndexMap(featureSectionKeys,
          featureShardIdToInterceptMap.getOrElse(shardId, true))
        (shardId, featureMap)
      }
    featureShardIdToFeatureMapMap.foreach { case (shardId, featureMap) =>
      logger.debug(s"Feature shard ID: $shardId, number of features: ${featureMap.size}")
    }
    featureShardIdToFeatureMapMap
  }

  /**
   * Builds a GAME data set according to input data configuration
   *
   * @param featureShardIdToFeatureMapMap a map of shard id to feature map
   * @return the prepared GAME data set
   */
  protected def prepareGameDataSet(featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]])
  : (RDD[(Long, String)], RDD[(Long, GameDatum)]) = {

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

    val gameDataSetWithUIDs = DataProcessingUtils.getGameDataSetWithUIDFromGenericRecords(
      recordsWithUniqueId,
      featureShardIdToFeatureSectionKeysMap,
      featureShardIdToFeatureMapMap,
      randomEffectIdSet,
      isResponseRequired = false)
      .partitionBy(globalDataPartitioner)
      .setName("Game data set with UIDs for scoring")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val gameDataSet = gameDataSetWithUIDs.mapValues(_._1)
    val uids = gameDataSetWithUIDs.mapValues(_._2)

    // Log some simple summary info on the Game data set
    logger.debug(s"Summary for the GAME data set")
    val numSamples = gameDataSetWithUIDs.count()
    logger.debug(s"numSamples: $numSamples")
    randomEffectIdSet.foreach { randomEffectId =>
      val numSamplesStats = gameDataSet.map { case (_, gameData) =>
        val individualId = gameData.randomEffectIdToIndividualIdMap(randomEffectId)
        (individualId, 1)
      }
        .reduceByKey(_ + _)
        .values
        .stats()
      logger.debug(s"numSamples for $randomEffectId: $numSamplesStats")
    }
    (uids, gameDataSet)
  }

  /**
   * Score the game data set with the game model
   * @param featureShardIdToFeatureMapMap a map of shard id to feature map
   * @param gameDataSet the game data set
   * @return the scores
   */
  //todo: make the number of files written to HDFS to be configurable
  protected def scoreGameDataSet(
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      gameDataSet: RDD[(Long, GameDatum)]): KeyValueScore = {

    val gameModel = ModelProcessingUtils.loadGameModelFromHDFS(featureShardIdToFeatureMapMap, gameModelInputDir,
      sparkContext)

    logger.debug(s"Loaded game model summary:\n${gameModel.toSummaryString}")

    val scores = gameModel.score(gameDataSet)
    val offsets = new KeyValueScore(gameDataSet.mapValues(_.offset))
    scores + offsets
  }

  /**
   * Run the driver
   */
  def run(): Unit = {

    // Process the output directory upfront and potentially fail the job early
    IOUtils.processOutputDir(outputDir, deleteOutputDirIfExists, sparkContext.hadoopConfiguration)

    var startTime = System.nanoTime()
    val featureShardIdToFeatureMapMap = prepareFeatureMaps()
    val initializationTime = (System.nanoTime() - startTime) * 1e-9
    logger.info(s"Time elapsed after preparing feature maps: $initializationTime (s)\n")

    startTime = System.nanoTime()
    val (uids, gameDataSet) = prepareGameDataSet(featureShardIdToFeatureMapMap)
    val gameDataSetPreparationTime = (System.nanoTime() - startTime) * 1e-9
    logger.info(s"Time elapsed after game data set preparation: $gameDataSetPreparationTime (s)\n")

    startTime = System.nanoTime()
    val scores = scoreGameDataSet(featureShardIdToFeatureMapMap, gameDataSet)
    val scoredItems = uids.join(scores.scores).map { case (_, (uid, score)) => ScoredItem(uid, score) }
    scoredItems.setName("Scored items").persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val numScoredItems = scoredItems.count()
    logger.info(s"Number of scored items: $numScoredItems (s)\n")
    val scoresDir = new Path(outputDir, Driver.SCORES).toString
    val scoredItemsToBeSaved =
      if (numOutputFilesForScores > 0 && numOutputFilesForScores != scoredItems.partitions.length) {
        scoredItems.coalesce(numOutputFilesForScores)
      } else {
        scoredItems
      }

    Utils.deleteHDFSDir(scoresDir, hadoopConfiguration)
    ScoreProcessingUtils.saveScoredItemsToHDFS(scoredItemsToBeSaved, modelId = "", scoresDir)
    val scoringTime = (System.nanoTime() - startTime) * 1e-9
    logger.info(s"Time elapsed scoring and writing scores to HDFS: $scoringTime (s)\n")

    val postprocessingTime = (System.nanoTime() - startTime) * 1e-9
    logger.info(s"Time elapsed after evaluation: $postprocessingTime (s)\n")
  }
}

object Driver {
  protected[scoring] val SCORES = "scores"
  protected[scoring] val LOGS = "logs"

  /**
   * Main entry point
   */
  def main(args: Array[String]): Unit = {

    val startTime = System.nanoTime()

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

      val timeElapsed = (System.nanoTime() - startTime) * 1e-9 / 60
      logger.info(s"Overall time elapsed $timeElapsed minutes")

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
