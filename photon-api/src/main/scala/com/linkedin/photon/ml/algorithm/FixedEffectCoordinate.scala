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
package com.linkedin.photon.ml.algorithm

import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Constants
import com.linkedin.photon.ml.Types.FeatureShardId
import com.linkedin.photon.ml.constants.DataConst
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.function.DistributedObjectiveFunction
import com.linkedin.photon.ml.model.{DatumScoringModel, FixedEffectModel}
import com.linkedin.photon.ml.optimization.{DistributedOptimizationProblem, FixedEffectOptimizationTracker, OptimizationTracker}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.VectorUtils

/**
 * The optimization problem coordinate for a fixed effect model.
 *
 * @tparam Objective The type of objective function used to solve the fixed effect optimization problem
 * @param rawData The raw training data
 * @param optimizationProblem The fixed effect optimization problem
 * @param inputColumnsNames
 */
protected[ml] class FixedEffectCoordinate[Objective <: DistributedObjectiveFunction](
    rawData: DataFrame,
    optimizationProblem: DistributedOptimizationProblem[Objective],
    featureShardId: FeatureShardId,
    inputColumnsNames: InputColumnsNames)
  extends Coordinate {

  var dataset: DataFrame =
    rawData
      .select(Constants.UNIQUE_SAMPLE_ID, featureShardId, inputColumnsNames(InputColumnsNames.RESPONSE))
      .withColumn(inputColumnsNames(InputColumnsNames.OFFSET), lit(0.0))


  override protected def updateDataset(scores: CoordinateDataScores) = {
      dataset = scores.scores
        .join(rawData, Constants.UNIQUE_SAMPLE_ID)
        .withColumn(inputColumnsNames(InputColumnsNames.OFFSET),
          col(inputColumnsNames(InputColumnsNames.OFFSET)) + col(DataConst.SCORE))
  }


  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(model: DatumScoringModel): (DatumScoringModel, OptimizationTracker) =
    model match {
      case fixedEffectModel: FixedEffectModel =>
        FixedEffectCoordinate.trainModel(
          dataset,
          optimizationProblem,
          featureShardId,
          Some(fixedEffectModel))

      case _ =>
        throw new UnsupportedOperationException(
          s"Training model of type ${model.getClass} in ${this.getClass} is not supported")
    }


  /**
   * Compute scores for the coordinate dataset using the given model.
   *
   * @param model The input model
   * @return The dataset scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = model match {

    case fixedEffectModel: FixedEffectModel =>
      FixedEffectCoordinate.score(dataset, fixedEffectModel, featureShardId)

    case _ =>
      throw new UnsupportedOperationException(
        s"Scoring with model of type ${model.getClass} in ${this.getClass} is not supported")
  }


  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A (updated model, optimization state tracking information) tuple
   */
  override protected def trainModel(): (DatumScoringModel, OptimizationTracker) =
    FixedEffectCoordinate.trainModel(dataset, optimizationProblem, featureShardId, None)
}

object FixedEffectCoordinate {

  /**
   * Train a new [[FixedEffectModel]] (i.e. run model optimization).
   *
   * @param dataset The training dataset
   * @param optimizationProblem The optimization problem
   * @param featureShardId The ID of the feature shard for the training data
   * @param initialFixedEffectModelOpt An optional existing [[FixedEffectModel]] to use as a starting point for
   *                                   optimization
   * @return A new [[FixedEffectModel]]
   */
  private def trainModel[Function <: DistributedObjectiveFunction](
    dataset: DataFrame,
    optimizationProblem: DistributedOptimizationProblem[Function],
    featureShardId: FeatureShardId,
    initialFixedEffectModelOpt: Option[FixedEffectModel]): (FixedEffectModel, OptimizationTracker) = {

    val rdd = dataset
      .rdd
      .map { row =>
        val features = row.getAs[SparkVector](0)
        val label = row.getDouble(1)

        LabeledPoint(label, VectorUtils.mlToBreeze(features))
      }
    rdd.persist(StorageLevel.MEMORY_ONLY)

    val (glm, stateTracker) = initialFixedEffectModelOpt
      .map { initialFixedEffectModel =>
        optimizationProblem.runWithSampling(rdd, initialFixedEffectModel.model)
      }
      .getOrElse(optimizationProblem.runWithSampling(rdd))

    rdd.unpersist()

    (new FixedEffectModel(SparkSession.builder.getOrCreate.sparkContext.broadcast(glm), featureShardId),
      new FixedEffectOptimizationTracker(stateTracker))
  }

  /**
   * Compute scores given a training dataset and a fixed effect model
   *
   * @param dataset The dataset to score
   * @param fixedEffectModel  The model used to score the dataset
   * @param featureShardId The ID of the feature shard for the training data
   * @return The computed scores
   */
  def score(dataset: DataFrame, fixedEffectModel: FixedEffectModel, featureShardId: FeatureShardId): CoordinateDataScores = {
    val cofs = VectorUtils.breezeToMl(fixedEffectModel.model.coefficients.means)
    val scores = dataset
      .withColumn(DataConst.SCORE, GeneralizedLinearModel.scoreUdf(lit(cofs), col(featureShardId)))
      .select(Constants.UNIQUE_SAMPLE_ID, DataConst.SCORE)
    new CoordinateDataScores(scores)
  }
}
