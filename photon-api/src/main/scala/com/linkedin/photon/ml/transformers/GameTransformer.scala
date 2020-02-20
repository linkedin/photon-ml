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
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, count, monotonically_increasing_id}
import org.apache.spark.storage.StorageLevel
import org.slf4j.Logger

import com.linkedin.photon.ml.Types.{REType, UniqueSampleId}
import com.linkedin.photon.ml.constants.DataConst
import com.linkedin.photon.ml.data.InputColumnsNames
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
  def transform(data: DataFrame): DataFrame = {

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
      data.withColumn(DataConst.ID, monotonically_increasing_id)
    }

    if (getOrDefault(logDataAndModelStats)) {
      logGameDataset(gameDataset, randomEffectTypes)
      logger.debug(s"GAME model summary:\n${getRequiredParam(model).toSummaryString}")
    }

    val storageLevel = if (getOrDefault(spillScoresToDisk)) {
      StorageLevel.MEMORY_AND_DISK
    } else {
      StorageLevel.MEMORY_ONLY
    }
    val gameDataWithScores = Timed("Computing scores") {
      getRequiredParam(model).score(gameDataset)
    }
    gameDataWithScores.persist(storageLevel)

    Timed("Evaluating scores") {
      get(validationEvaluators).foreach(
        _.foreach { evaluatorType =>
          val evaluationMetricValue = evaluateScores(evaluatorType, gameDataWithScores)
          logger.info(s"Evaluation metric value on scores with $evaluatorType: $evaluationMetricValue")
        })
    }

    gameDataWithScores
  }


  /**
   * Log some simple summary statistics for the GAME dataset.
   *
   * @param gameDataset The GAME dataset
   * @param randomEffectTypes The set of unique identifier fields used by the random effects of the model
   */
  private def logGameDataset(gameDataset: DataFrame, randomEffectTypes: Set[REType]): Unit = {

    val numSamples = gameDataset.count()

    logger.debug(s"Summary for the GAME dataset")
    logger.debug(s"numSamples: $numSamples")

    randomEffectTypes.foreach { idTag =>
      val numSamplesStats = gameDataset
          .groupBy(idTag).agg(count("*").alias("cnt"))
        .describe("cnt")
        .collect()
        .map(t => t.getString(0) + "\t" + t.getDouble(1) + "\t" + t.getDouble(2))
        .mkString("\n")

      logger.debug(s"numSamples for $idTag: $numSamplesStats")
    }
  }


  /**
   * Evaluate the computed scores with the given evaluator type.
   *
   * @param evaluatorType The evaluator type
   * @param gameDatasetWithscores The GAME dataset
   * @return The evaluation metric
   */
  protected def evaluateScores(
      evaluatorType: EvaluatorType,
      gameDatasetWithscores: DataFrame): Double = {

    val evaluator = EvaluatorFactory.buildEvaluator(evaluatorType, gameDatasetWithscores)

    val offset = inputColumnNames(InputColumnsNames.OFFSET)
    val response = inputColumnNames(InputColumnsNames.RESPONSE)
    val weight = inputColumnNames(InputColumnsNames.WEIGHT)
    evaluator match {
      case se: SingleEvaluator =>
        val scoresRDD = gameDatasetWithscores
            .select(col(DataConst.SCORE) + col(offset), response, weight)
            .rdd.map (row => (row.getDouble(0), row.getDouble(1), row.getDouble(2)))

        se.evaluate(scoresRDD)

      case me: MultiEvaluator =>
        val scoresRDD = gameDatasetWithscores
        .select(col(DataConst.ID), col(DataConst.SCORE) + col(offset), response, weight)
        .rdd.map (row => (row.getAs[UniqueSampleId](0), (row.getDouble(1), row.getDouble(2), row.getDouble(3))))

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
