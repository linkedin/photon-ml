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

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.avro.AvroUtils
import com.linkedin.photon.ml.avro.data.{DataProcessingUtils, NameAndTermFeatureSetContainer, NameAndTerm}
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.evaluation.{RMSEEvaluator, BinaryClassificationEvaluator}
import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.supervised.TaskType._
import com.linkedin.photon.ml.util._


/**
 * Driver for GAME full model scoring
 *
 * @author xazhang
 */
class Driver(val params: Params, val sparkContext: SparkContext, val logger: PhotonLogger) {

  import params._

  protected val parallelism: Int = sparkContext.getConf.get("spark.default.parallelism",
    s"${sparkContext.getExecutorStorageStatus.length * 3}").toInt
  protected val hadoopConfiguration = sparkContext.hadoopConfiguration

  protected val isAddingIntercept = true

  /**
   * Builds feature name-and-term to index maps according to configuration
   *
   * @return a map of shard id to feature map
   */
  protected def prepareFeatureMaps(): Map[String, Map[NameAndTerm, Int]] = {

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
      logger.debug(s"Feature shard ID: $shardId, number of features: ${featureMap.size}")
    }
    featureShardIdToFeatureMapMap
  }

  /**
   * Builds a GAME dataset according to input data configuration
   *
   * @param featureShardIdToFeatureMapMap a map of shard id to feature map
   * @return the prepared GAME dataset
   */
  protected def prepareGameDataSet(featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]])
  : RDD[(Long, GameDatum)] = {

    val recordsPath = dateRangeOpt match {
      case Some(dateRange) =>
        val Array(startDate, endDate) = dateRange.split("-")
        IOUtils.getInputPathsWithinDateRange(inputDirs, startDate, endDate, hadoopConfiguration, errorOnMissing = false)
      case None => inputDirs.toSeq
    }
    logger.debug(s"Avro records paths:\n${recordsPath.mkString("\n")}")
    val records = AvroUtils.readAvroFiles(sparkContext, recordsPath, parallelism)
    val globalDataPartitioner = new LongHashPartitioner(records.partitions.length)

    val gameDataSet = DataProcessingUtils.getGameDataSetFromGenericRecords(records,
      featureShardIdToFeatureSectionKeysMap, featureShardIdToFeatureMapMap, randomEffectIdSet)
        .partitionBy(globalDataPartitioner)
        .setName("Scoring Game data set")
        .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)

    // Log some simple summary info on the Game data set
    logger.debug(s"Summary for the validating Game data set")
    val numSamples = gameDataSet.count()
    logger.debug(s"numSamples: $numSamples")
    val responseSum = gameDataSet.values.map(_.response).sum()
    logger.debug(s"responseSum: $responseSum")
    val weightSum = gameDataSet.values.map(_.weight).sum()
    logger.debug(s"weightSum: $weightSum")
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
      logger.debug(s"numSamplesStats for $randomEffectId: $numSamplesStats")
      logger.debug(s"responseSumStats for $randomEffectId: $responseSumStats")
    }

    gameDataSet
  }

  /**
   * Score and write results to HDFS
   * @param featureShardIdToFeatureMapMap a map of shard id to feature map
   * @param gameDataSet the input dataset
   * @return the scores
   */
  //todo: make the number of files written to HDFS to be configurable
  protected def scoreAndWriteScoreToHDFS(
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      gameDataSet: RDD[(Long, GameDatum)]): RDD[(Long, Double)] = {

    val gameModel = ModelProcessingUtils.loadGameModelFromHDFS(featureShardIdToFeatureMapMap, gameModelInputDir,
      sparkContext)

    gameModel.foreach {
      case rddLike: RDDLike => rddLike.persistRDD(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
      case _ =>
    }
    logger.debug(s"Loaded game model summary:\n${gameModel.map(_.toSummaryString).mkString("\n")}")

    val scores = gameModel.map(_.score(gameDataSet)).reduce(_ + _).scores
        .setName("Scores").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val scoredItems = scores.join(gameDataSet).map { case (_, (score, gameData)) =>
      val ids = gameData.randomEffectIdToIndividualIdMap.values
      val label = gameData.response
      ScoredItem(ids, score, label)
    }.setName("Scored item").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val numScoredItems = scoredItems.count()
    val scoresDir = new Path(outputDir, Driver.SCORES).toString
    Utils.deleteHDFSDir(scoresDir, hadoopConfiguration)
    // Should always materialize the scoredItems first (e.g., count()) before the coalesce happens
    scoredItems.coalesce(numFiles).saveAsTextFile(scoresDir)
    logger.debug(s"Number of scored items: $numScoredItems")

    scores
  }

  /**
   * Evaluate and log metrics for the GAME dataset
   *
   * @param gameDataSet the input dataset
   * @param scores the scores
   */
  protected def evaluateAndLog(gameDataSet: RDD[(Long, GameDatum)], scores: RDD[(Long, Double)]) {

    val validatingLabelAndOffsets = gameDataSet.mapValues(gameData => (gameData.response, gameData.offset))
    val metric =  taskType match {
      case LOGISTIC_REGRESSION =>
        new BinaryClassificationEvaluator(validatingLabelAndOffsets).evaluate(scores)
      case LINEAR_REGRESSION =>
        val validatingLabelAndOffsetAndWeights = validatingLabelAndOffsets.mapValues { case (label, offset) =>
          (label, offset, 1.0)
        }
        new RMSEEvaluator(validatingLabelAndOffsetAndWeights).evaluate(scores)
      case _ =>
        throw new UnsupportedOperationException(s"Task type: $taskType is not supported to create validating " +
            s"evaluator")
    }
    logger.info(s"Evaluation metric: $metric")
  }

  /**
   * Run the driver
   */
  def run(): Unit = {

    var startTime = System.nanoTime()
    val featureShardIdToFeatureMapMap = prepareFeatureMaps()
    val initializationTime = (System.nanoTime() - startTime) * 1e-9
    logger.info(s"Time elapsed after preparing feature maps: $initializationTime (s)\n")

    startTime = System.nanoTime()
    val gameDataSet = prepareGameDataSet(featureShardIdToFeatureMapMap)
    val gameDataSetPreparationTime = (System.nanoTime() - startTime) * 1e-9
    logger.info(s"Time elapsed after game data set preparation: $gameDataSetPreparationTime (s)\n")

    startTime = System.nanoTime()
    val scores = scoreAndWriteScoreToHDFS(featureShardIdToFeatureMapMap, gameDataSet)
    val scoringTime = (System.nanoTime() - startTime) * 1e-9
    logger.info(s"Time elapsed scoring and writing scores to HDFS: $scoringTime (s)\n")

    startTime = System.nanoTime()
    evaluateAndLog(gameDataSet, scores)
    val postprocessingTime = (System.nanoTime() - startTime) * 1e-9
    logger.info(s"Time elapsed after evaluation: $postprocessingTime (s)\n")
  }
}

object Driver {
  private val SCORES = "scores"
  private val LOGS = "logs"

  /**
   * Main entry point
   */
  def main(args: Array[String]): Unit = {

    val startTime = System.nanoTime()

    val params = Params.parseFromCommandLine(args)
    import params._

    val sc = SparkContextConfiguration.asYarnClient(applicationName, useKryo = true)

    val logsDir = new Path(outputDir, LOGS).toString
    Utils.createHDFSDir(logsDir, sc.hadoopConfiguration)
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
