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
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

import com.linkedin.photon.ml.{DataValidationType, SparkContextConfiguration, TaskType}
import com.linkedin.photon.ml.Types.{FeatureShardId, UniqueSampleId}
import com.linkedin.photon.ml.cli.game.GameDriver
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.avro._
import com.linkedin.photon.ml.data.scoring.ModelDataScores
import com.linkedin.photon.ml.data.{DataValidators, GameConverters, GameDatum, InputColumnsNames}
import com.linkedin.photon.ml.evaluation.{EvaluatorFactory, EvaluatorType, MultiEvaluatorType}
import com.linkedin.photon.ml.index.IndexMapLoader
import com.linkedin.photon.ml.io.scopt.game.ScoptGameScoringParametersParser
import com.linkedin.photon.ml.model.RandomEffectModel
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

  val modelInputDirectory: Param[Path] = ParamUtils.createParam(
    "model input directory",
    "Path to directory containing model to use for scoring.")

  val randomEffectTypes: Param[Set[String]] = ParamUtils.createParam(
    "random effect types",
    "The set of random effect types used by the random effect models.")

  val modelId: Param[String] = ParamUtils.createParam(
    "model id",
    "ID to tag scores with.")

  val logDataAndModelStats: Param[Boolean] = ParamUtils.createParam(
    "log data and model stats",
    "Whether to log data set and model statistics (can be time-consuming for very large data sets).")

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
  // Params functions
  //

  /**
   * Check that all required parameters have been set and validate interactions between parameters.
   */
  override def validateParams(paramMap: ParamMap = extractParamMap): Unit = {

    super.validateParams(paramMap)

    // Just need to check that these parameters are explicitly set
    paramMap(modelInputDirectory)
  }

  /**
   * Set default values for parameters that have them.
   */
  private def setDefaultParams(): Unit = {

    setDefault(inputColumnNames, InputColumnsNames())
    setDefault(overrideOutputDirectory, false)
    setDefault(dataValidation, DataValidationType.VALIDATE_DISABLED)
    setDefault(logDataAndModelStats, false)
    setDefault(spillScoresToDisk, false)
    setDefault(logLevel, PhotonLogger.LogLevelInfo)
    setDefault(applicationName, DEFAULT_APPLICATION_NAME)
  }

  /**
   * Clear all set parameters.
   */
  def clear(): Unit = params.foreach(clear)

  //
  // Scoring driver functions
  //

  /**
   * Run the driver.
   */
  protected[scoring] def run(): Unit = {
    val timer = new Timer

    // Process the output directory upfront and potentially fail the job early
    IOUtils.processOutputDir(
      getRequiredParam(rootOutputDirectory),
      getOrDefault(overrideOutputDirectory),
      sc.hadoopConfiguration)

    timer.start()
    val featureShardIdToFeatureMapMap = prepareFeatureMaps()
    timer.stop()
    logger.info(s"Time elapsed after preparing feature maps: ${timer.durationSeconds} (s)\n")

    val dataFrame = Timed("Read data") {
      readDataFrame(featureShardIdToFeatureMapMap)
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

    timer.start()
    val gameDataSet = prepareGameDataSet(dataFrame)
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
    get(evaluators).foreach(_.foreach { evaluatorType =>
      val evaluationMetricValue = evaluateScores(evaluatorType, scores, gameDataSet)
      logger.info(s"Evaluation metric value on scores with $evaluatorType: $evaluationMetricValue")
    })
    timer.stop()
    logger.info(s"Time elapsed after evaluating scores: ${timer.durationSeconds} (s)\n")
  }

  /**
   * Reads AVRO input data into a [[DataFrame]].
   *
   * @param featureShardIdToFeatureMapLoader A map of shard id to feature map loader
   * @return A [[DataFrame]] of input data
   */
  protected def readDataFrame(featureShardIdToFeatureMapLoader: Map[FeatureShardId, IndexMapLoader]): DataFrame = {

    val parallelism = sc.getConf.get("spark.default.parallelism", s"${sc.getExecutorStorageStatus.length * 3}").toInt

    // Handle date range input
    val dateRangeOpt = IOUtils.resolveRange(get(inputDataDateRange), get(inputDataDaysRange))
    val recordsPath = pathsForDateRange(getRequiredParam(inputDataDirectories), dateRangeOpt)

    logger.debug(s"Input records paths:\n${recordsPath.mkString("\n")}")

    new AvroDataReader(sc).readMerged(
      recordsPath.map(_.toString),
      featureShardIdToFeatureMapLoader.toMap,
      getRequiredParam(featureShardConfigurations).mapValues(_.featureBags).map(identity),
      parallelism)
  }

  /**
   * Builds a GAME data set according to input data configuration.
   *
   * @param dataFrame A [[DataFrame]] of raw input data
   * @return The prepared GAME data set
   */
  protected def prepareGameDataSet(dataFrame: DataFrame): RDD[(UniqueSampleId, GameDatum)] = {

    val parallelism = sc.getConf.get("spark.default.parallelism", s"${sc.getExecutorStorageStatus.length * 3}").toInt
    val partitioner = new LongHashPartitioner(parallelism)
    val idTagSet: Set[String] = get(randomEffectTypes).getOrElse(Set()) ++
      get(evaluators).map(MultiEvaluatorType.getMultiEvaluatorIdTags).getOrElse(Seq())
    val gameDataSet = GameConverters
      .getGameDataSetFromDataFrame(
        dataFrame,
        getRequiredParam(featureShardConfigurations).keys.toSet,
        idTagSet,
        isResponseRequired = false,
        getOrDefault(inputColumnNames))
      .partitionBy(partitioner)
      .setName("Game data set with UIDs for scoring")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    if (getOrDefault(logDataAndModelStats)) {
      logGameDataSet(gameDataSet)
    }

    gameDataSet
  }

  /**
   * Log some simple summary statistics for the GAME data set.
   *
   * @param gameDataSet The GAME data set
   */
  private def logGameDataSet(gameDataSet: RDD[(UniqueSampleId, GameDatum)]): Unit = {

    val numSamples = gameDataSet.count()

    logger.debug(s"Summary for the GAME data set")
    logger.debug(s"numSamples: $numSamples")

    get(randomEffectTypes).foreach(_.foreach { idTag =>
      val numSamplesStats = gameDataSet
        .map { case (_, gameData) =>
          val idValue = gameData.idTagToValueMap(idTag)
          (idValue, 1)
        }
        .reduceByKey(_ + _)
        .values
        .stats()

      logger.debug(s"numSamples for $idTag: $numSamplesStats")
    })
  }

  /**
   * Load the GAME model and score the GAME data set.
   *
   * @param featureShardIdToIndexMapLoader A map of feature shard id to feature map loader
   * @param gameDataSet The GAME data set
   * @return The scores
   */
  protected def scoreGameDataSet(
      featureShardIdToIndexMapLoader: Map[FeatureShardId, IndexMapLoader],
      gameDataSet: RDD[(UniqueSampleId, GameDatum)]): ModelDataScores = {

    // Load the model from HDFS, ignoring the feature index loader
    val (gameModel, _) = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      getRequiredParam(modelInputDirectory),
      StorageLevel.VERY_FREQUENT_REUSE_RDD_STORAGE_LEVEL,
      Some(featureShardIdToIndexMapLoader))

    if (getOrDefault(logDataAndModelStats)) {
      logger.debug(s"Loaded game model summary:\n${gameModel.toSummaryString}")
    }

    // Need to split these calls to keep correct return type
    val scores = gameModel.score(gameDataSet)
    val storageLevel = if (getOrDefault(spillScoresToDisk)) {
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

    if (getOrDefault(logDataAndModelStats)) {
      // Persist scored items here since we introduce multiple passes
      scoredItems.setName("Scored items").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)

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
      gameDataSet: RDD[(UniqueSampleId, GameDatum)]): Double = {

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

    val params: ParamMap = ScoptGameScoringParametersParser.parseFromCommandLine(args)
    params.toSeq.foreach(set)

    sc = SparkContextConfiguration.asYarnClient(getOrDefault(applicationName), useKryo = true)
    logger = new PhotonLogger(getRequiredParam(rootOutputDirectory), sc)
    logger.setLogLevel(getOrDefault(logLevel))

    try {

      run()

      timer.stop()
      logger.info(s"Overall time elapsed ${timer.durationMinutes} minutes")

    } catch { case e: Exception =>

      logger.error("Failure while running the driver", e)
      throw e

    } finally {

      logger.close()
      sc.stop()
    }
  }
}
