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

import org.apache.commons.cli.MissingArgumentException
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.{Constants, DataValidationType, SparkSessionConfiguration, TaskType}
import com.linkedin.photon.ml.Types.FeatureShardId
import com.linkedin.photon.ml.cli.game.GameDriver
import com.linkedin.photon.ml.data.avro._
import com.linkedin.photon.ml.data.{DataValidators, InputColumnsNames}
import com.linkedin.photon.ml.index.IndexMapLoader
import com.linkedin.photon.ml.io.scopt.game.ScoptGameScoringParametersParser
import com.linkedin.photon.ml.model.RandomEffectModel
import com.linkedin.photon.ml.transformers.GameTransformer
import com.linkedin.photon.ml.util._

/**
 * Driver for GAME full model scoring.
 */
object GameScoringDriver extends GameDriver {

  //
  // Members
  //

  private val DEFAULT_APPLICATION_NAME = "GAME-Scoring"

  protected[scoring] implicit var logger: PhotonLogger = _
  protected[scoring] var sc: SparkContext = _

  val SCORES_DIR = "scores"

  //
  // Parameters
  //

  val modelId: Param[String] = ParamUtils.createParam(
    "model id",
    "ID to tag scores with.")

  val logDataAndModelStats: Param[Boolean] = ParamUtils.createParam(
    "log data and model stats",
    "Whether to log dataset and model statistics (can be time-consuming for very large datasets).")

  val spillScoresToDisk: Param[Boolean] = ParamUtils.createParam(
    "spill scores to disk",
    "Whether to spill data to disk when memory is full (more CPU intensive, but prevents recomputation if memory " +
      "blocks are evicted). Useful for very large scoring tasks.")

  //
  // Initialize object
  //

  setDefaultParams()

  //
  // Params trait extensions
  //

  /**
   * Copy function has no meaning for Driver object. Add extra parameters to params and return.
   *
   * @param extra Additional parameters which should overwrite the values being copied
   * @return This object
   */
  override def copy(extra: ParamMap): Params = {

    extra.toSeq.foreach(set)

    this
  }

  //
  // PhotonParams trait extensions
  //

  /**
   * Set default values for parameters that have them.
   */
  override protected def setDefaultParams(): Unit = {

    setDefault(inputColumnNames, InputColumnsNames())
    setDefault(overrideOutputDirectory, false)
    setDefault(dataValidation, DataValidationType.VALIDATE_DISABLED)
    setDefault(logDataAndModelStats, false)
    setDefault(spillScoresToDisk, false)
    setDefault(logLevel, PhotonLogger.LogLevelInfo)
    setDefault(applicationName, DEFAULT_APPLICATION_NAME)
    setDefault(timeZone, Constants.DEFAULT_TIME_ZONE)
  }

  /**
   * Check that all required parameters have been set and validate interactions between parameters.
   *
   * @note In Spark, interactions between parameters are checked by
   *       [[org.apache.spark.ml.PipelineStage.transformSchema()]]. Since we do not use the Spark pipeline API in
   *       Photon-ML, we need to have this function to check the interactions between parameters.
   * @throws MissingArgumentException if a required parameter is missing
   * @throws IllegalArgumentException if a required parameter is missing or a validation check fails
   * @param paramMap The parameters to validate
   */
  override def validateParams(paramMap: ParamMap = extractParamMap): Unit = {

    super.validateParams(paramMap)

    // Just need to check that these parameters are explicitly set
    paramMap(modelInputDirectory)
  }

  //
  // Scoring driver functions
  //

  /**
   * Run the driver.
   */
  protected[scoring] def run(): Unit = {

    validateParams()

    // Process the output directory upfront and potentially fail the job early
    IOUtils.processOutputDir(
      getRequiredParam(rootOutputDirectory),
      getOrDefault(overrideOutputDirectory),
      sc.hadoopConfiguration)

    val featureShardIdToIndexMapLoaderMapOpt = Timed("Prepare features") {
      prepareFeatureMaps()
    }

    val (dataFrame, featureShardIdToIndexMapLoaderMap) = Timed("Read data") {
      readDataFrame(featureShardIdToIndexMapLoaderMapOpt)
    }

    // TODO: The model training task should be read from the metadata. For now, hardcode to LINEAR_REGRESSION, since
    // TODO: that has the least strict checks.
    Timed("Validate data") {
      DataValidators.sanityCheckDataFrameForScoring(
        dataFrame,
        getOrDefault(dataValidation),
        getOrDefault(inputColumnNames),
        getRequiredParam(featureShardConfigurations).keySet,
        get(evaluators).map(_ => TaskType.LINEAR_REGRESSION))
    }

    val gameModel = Timed("Load model") {
      ModelProcessingUtils.loadGameModelFromHDFS(
        sc,
        getRequiredParam(modelInputDirectory),
        StorageLevel.MEMORY_ONLY,
        featureShardIdToIndexMapLoaderMap)
    }

    val gameTransformer = Timed("Setup transformer") {
      val transformer = new GameTransformer(sc, logger)
        .setModel(gameModel)
        .setLogDataAndModelStats(getOrDefault(logDataAndModelStats))
        .setSpillScoresToDisk(getOrDefault(spillScoresToDisk))

      get(inputColumnNames).foreach(transformer.setInputColumnNames)
      get(evaluators).foreach(transformer.setValidationEvaluators)

      transformer
    }

    val scores = Timed("Score data") {
      gameTransformer.transform(dataFrame)
    }

    gameModel.toMap.foreach {
      case (_, model: RandomEffectModel) => model.unpersistRDD()
      case _ =>
    }

    Timed("Save scores") {
      saveScoresToHDFS(scores)
    }
  }

  /**
   * Reads AVRO input data into a [[DataFrame]].
   *
   * @param featureShardIdToIndexMapLoaderMapOpt An optional map of shard id to feature map loader
   * @return A ([[DataFrame]] of input data, map of shard id to feature map loader) pair
   */
  protected def readDataFrame(
      featureShardIdToIndexMapLoaderMapOpt: Option[Map[FeatureShardId, IndexMapLoader]])
    : (DataFrame, Map[FeatureShardId, IndexMapLoader]) = {

    val parallelism = sc.getConf.get("spark.default.parallelism", s"${sc.getExecutorStorageStatus.length * 3}").toInt

    // Handle date range input
    val dateRangeOpt = IOUtils.resolveRange(get(inputDataDateRange), get(inputDataDaysRange), getOrDefault(timeZone))
    val recordsPaths = pathsForDateRange(getRequiredParam(inputDataDirectories), dateRangeOpt)

    logger.debug(s"Input records paths:\n${recordsPaths.mkString("\n")}")

    new AvroDataReader().readMerged(
      recordsPaths.map(_.toString),
      featureShardIdToIndexMapLoaderMapOpt,
      getRequiredParam(featureShardConfigurations),
      parallelism)
  }

  /**
   * Save the computed scores to HDFS with auxiliary info.
   *
   * @param scores The computed scores
   */
  protected def saveScoresToHDFS(scores: ModelDataScores): Unit = {

    // Take the offset information into account when writing the scores to HDFS
    val scoredItems = scores.scoresRdd.map { case (_, scoredGameDatum) =>
      ScoredItem(
        scoredGameDatum.score + scoredGameDatum.offset,
        Some(scoredGameDatum.response),
        Some(scoredGameDatum.weight),
        scoredGameDatum.idTagToValueMap)
    }

    if (getOrDefault(logDataAndModelStats)) {
      // Persist scored items here since we introduce multiple passes
      scoredItems.setName("Scored items").persist(StorageLevel.MEMORY_AND_DISK)

      val numScoredItems = scoredItems.count()
      logger.info(s"Number of scored items to be written to HDFS: $numScoredItems \n")
    }

    val scoredItemsToBeSaved = get(outputFilesLimit) match {
      case Some(limit) if limit < scoredItems.partitions.length => scoredItems.coalesce(getOrDefault(outputFilesLimit))
      case _ => scoredItems
    }
    val scoresDir = new Path(getRequiredParam(rootOutputDirectory), SCORES_DIR)

    ScoreProcessingUtils.saveScoredItemsToHDFS(scoredItemsToBeSaved, scoresDir.toString, get(modelId))
    scoredItems.unpersist()
  }

  /**
   * Entry point to the driver.
   *
   * @param args The command line arguments for the job
   */
  def main(args: Array[String]): Unit = {

    val params: ParamMap = ScoptGameScoringParametersParser.parseFromCommandLine(args)
    params.toSeq.foreach(set)

    sc = SparkSessionConfiguration.asYarnClient(getOrDefault(applicationName), useKryo = true).sparkContext
    logger = new PhotonLogger(new Path(getRequiredParam(rootOutputDirectory), LOGS_FILE_NAME), sc)
    logger.setLogLevel(getOrDefault(logLevel))

    try {
      Timed("Total time in scoring Driver")(run())

    } catch { case e: Exception =>
      logger.error("Failure while running the driver", e)
      throw e

    } finally {
      logger.close()
      sc.stop()
    }
  }
}
