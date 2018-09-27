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
package com.linkedin.photon.ml.transformers

import org.apache.commons.cli.MissingArgumentException
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel
import org.slf4j.Logger

import com.linkedin.photon.ml.Types.{FeatureShardId, REType, UniqueSampleId}
import com.linkedin.photon.ml.data.scoring.ModelDataScores
import com.linkedin.photon.ml.data.{GameConverters, GameDatum, InputColumnsNames}
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.model.{FixedEffectModel, GameModel, RandomEffectModel}
import com.linkedin.photon.ml.util._

/**
 * Scores input data using a [[GameModel]]. Plays a similar role to the [[org.apache.spark.ml.Model]].
 *
 * @param sc The spark context for the application
 * @param logger The logger instance for the application
 */
class GameTransformer(val sc: SparkContext, implicit val logger: Logger) extends PhotonParams {

  import GameTransformer._

  private implicit val parent: Identifiable = this

  override val uid: String = Identifiable.randomUID(GAME_TRANSFORMER_PREFIX)

  //
  // Parameters
  //

  val inputColumnNames: Param[InputColumnsNames] = ParamUtils.createParam[InputColumnsNames](
    "input column names",
    "A map of custom column names which replace the default column names of expected fields in the Avro input.")

  val model: Param[GameModel] = ParamUtils.createParam(
    "model",
    "The model to use for scoring input data.")

  val validationEvaluators: Param[Seq[EvaluatorType]] = ParamUtils.createParam(
    "validation evaluators",
    "A list of evaluators used to validate computed scores (Note: the first evaluator in the list is the one used " +
      "for model selection)",
    PhotonParamValidators.nonEmpty[Seq, EvaluatorType])

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
  // Parameter setters
  //

  def setInputColumnNames(value: InputColumnsNames): this.type = set(inputColumnNames, value)

  def setModel(value: GameModel): this.type = set(model, value)

  def setValidationEvaluators(value: Seq[EvaluatorType]): this.type = set(validationEvaluators, value)

  def setLogDataAndModelStats(value: Boolean): this.type = set(logDataAndModelStats, value)

  def setSpillScoresToDisk(value: Boolean): this.type = set(spillScoresToDisk, value)

  //
  // Params trait extensions
  //

  override def copy(extra: ParamMap): GameTransformer = {

    val copy = new GameTransformer(sc, logger)

    extractParamMap(extra).toSeq.foreach { paramPair =>
      copy.set(copy.getParam(paramPair.param.name), paramPair.value)
    }

    copy
  }

  //
  // PhotonParams trait extensions
  //

  /**
   * Set the default parameters.
   */
  override protected def setDefaultParams(): Unit = {

    setDefault(inputColumnNames, InputColumnsNames())
    setDefault(logDataAndModelStats, false)
    setDefault(spillScoresToDisk, false)
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

    // Just need to check that these parameters are explicitly set
    getRequiredParam(model)
  }

  //
  // GameTransformer functions
  //

  /**
   * Transform a [[DataFrame]] of data samples to include scores computed using the sample feature vectors and the
   * currently set model.
   *
   * @param data Input [[DataFrame]] of samples
   * @return Scored data samples
   */
  def transform(data: DataFrame): ModelDataScores = {

    validateParams()

    val randomEffectTypes = getRequiredParam(model)
      .toMap
      .values
      .flatMap {
        case rem: RandomEffectModel => Some(rem.randomEffectType)
        case _ => None
      }
      .toSet
    val featureShards = getRequiredParam(model)
      .toMap
      .values
      .map {
        case fem: FixedEffectModel => fem.featureShardId
        case rem: RandomEffectModel => rem.featureShardId
      }
      .toSet

    val gameDataset = Timed("Preparing GAME dataset") {
      prepareGameDataset(data, randomEffectTypes, featureShards)
    }

    if (getOrDefault(logDataAndModelStats)) {
      logGameDataset(gameDataset, randomEffectTypes)
      logger.debug(s"GAME model summary:\n${getRequiredParam(model).toSummaryString}")
    }

    val scores = Timed("Computing scores") {
      scoreGameDataset(gameDataset)
    }

    gameDataset.unpersist()

    Timed("Evaluating scores") {
      get(validationEvaluators).foreach(
        _.foreach { evaluatorType =>
          val evaluationMetricValue = evaluateScores(evaluatorType, gameDataset, scores)
          logger.info(s"Evaluation metric value on scores with $evaluatorType: $evaluationMetricValue")
        })
    }

    // TODO: Instead, we should merge the scores back into the DataFrame in a new column (at least optionally)

    scores
  }

  /**
   * Builds a GAME dataset according to input data configuration.
   *
   * @param dataFrame A [[DataFrame]] of raw input data
   * @param randomEffectTypes The set of unique identifier fields used by the random effects of the model
   * @param featureShards The set of feature shards used by the model
   * @return The prepared GAME dataset
   */
  protected def prepareGameDataset(
      dataFrame: DataFrame,
      randomEffectTypes: Set[REType],
      featureShards: Set[FeatureShardId]): RDD[(UniqueSampleId, GameDatum)] = {

    val parallelism = sc.getConf.get("spark.default.parallelism", s"${sc.getExecutorStorageStatus.length * 3}").toInt
    val partitioner = new LongHashPartitioner(parallelism)
    val idTagSet = randomEffectTypes ++
      get(validationEvaluators).map(MultiEvaluatorType.getMultiEvaluatorIdTags).getOrElse(Seq())
    val gameDataset = GameConverters
      .getGameDatasetFromDataFrame(
        dataFrame,
        featureShards,
        idTagSet,
        isResponseRequired = false,
        getOrDefault(inputColumnNames))
      .partitionBy(partitioner)
      .setName("Game dataset with UIDs for scoring")
      .persist(StorageLevel.DISK_ONLY)

    gameDataset
  }

  /**
   * Log some simple summary statistics for the GAME dataset.
   *
   * @param gameDataset The GAME dataset
   * @param randomEffectTypes The set of unique identifier fields used by the random effects of the model
   */
  private def logGameDataset(gameDataset: RDD[(UniqueSampleId, GameDatum)], randomEffectTypes: Set[REType]): Unit = {

    val numSamples = gameDataset.count()

    logger.debug(s"Summary for the GAME dataset")
    logger.debug(s"numSamples: $numSamples")

    randomEffectTypes.foreach { idTag =>
      val numSamplesStats = gameDataset
        .map { case (_, gameData) =>
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
   * Load the GAME model and score the GAME dataset.
   *
   * @param gameDataset The GAME dataset
   * @return The scores
   */
  protected def scoreGameDataset(gameDataset: RDD[(UniqueSampleId, GameDatum)]): ModelDataScores = {

    val storageLevel = if (getOrDefault(spillScoresToDisk)) {
      StorageLevel.MEMORY_AND_DISK
    } else {
      StorageLevel.MEMORY_ONLY
    }
    // Need to split these calls to keep correct return type
    val scores = getRequiredParam(model).score(gameDataset)
    scores.persistRDD(storageLevel).materialize()

    scores
  }

  /**
   * Evaluate the computed scores with the given evaluator type.
   *
   * @param evaluatorType The evaluator type
   * @param scores The computed scores
   * @param gameDataset The GAME dataset
   * @return The evaluation metric
   */
  protected def evaluateScores(
      evaluatorType: EvaluatorType,
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      scores: ModelDataScores): Double = {

    val evaluator = EvaluatorFactory.buildEvaluator(evaluatorType, gameDataset)

    evaluator match {
      case se: SingleEvaluator =>
        val scoresRDD = scores.scores.map { case (_, sGD) =>
          (sGD.score + sGD.offset, sGD.response, sGD.weight)
        }

        se.evaluate(scoresRDD)

      case me: MultiEvaluator =>
        val scoresRDD = scores.scores.mapValues(sGD => (sGD.score + sGD.offset, sGD.response, sGD.weight))

        me.evaluate(scoresRDD)

      case e =>
        throw new UnsupportedOperationException(s"Cannot process unknown Evaluator subtype: ${e.getClass}")
    }
  }
}

object GameTransformer {

  //
  // Constants
  //

  private val GAME_TRANSFORMER_PREFIX = "GameTransformer"
}
