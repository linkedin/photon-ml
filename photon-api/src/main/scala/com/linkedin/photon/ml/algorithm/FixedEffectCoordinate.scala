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
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.FeatureShardId
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.function.DistributedObjectiveFunction
import com.linkedin.photon.ml.model.{DatumScoringModel, FixedEffectModel}
import com.linkedin.photon.ml.optimization.{DistributedOptimizationProblem, FixedEffectOptimizationTracker, OptimizationTracker}
import com.linkedin.photon.ml.util.{ApiUtils, VectorUtils}

/**
 * The optimization problem coordinate for a fixed effect model.
 *
 * @tparam Objective The type of objective function used to solve the fixed effect optimization problem
 * @param dataset The raw training data
 * @param optimizationProblem The fixed effect optimization problem
 * @param inputColumnsNames
 */
protected[ml] class FixedEffectCoordinate[Objective <: DistributedObjectiveFunction](
    var dataset: DataFrame,
    optimizationProblem: DistributedOptimizationProblem[Objective],
    featureShardId: FeatureShardId,
    inputColumnsNames: InputColumnsNames)
  extends Coordinate {

  override protected def updateOffset(model: DatumScoringModel) = {
    model match {
      case fixedEffectModel: FixedEffectModel =>
        dataset = FixedEffectCoordinate.updateOffset(dataset, fixedEffectModel, featureShardId, inputColumnsNames)
      case _ =>
        throw new UnsupportedOperationException(s"Unsupported model type: ${model.modelType}")
    }
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
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A (updated model, optimization state tracking information) tuple
   */
  override protected def trainModel(): (DatumScoringModel, OptimizationTracker) =
    FixedEffectCoordinate.trainModel(dataset, optimizationProblem, featureShardId, None)

}

object FixedEffectCoordinate {

  def SCORE_FIELD = "fixed_score"

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
      .map ( initialFixedEffectModel =>
        optimizationProblem.runWithSampling(rdd, initialFixedEffectModel.model)
      )
      .getOrElse(optimizationProblem.runWithSampling(rdd))

    rdd.unpersist()

    (FixedEffectModel(glm, featureShardId), new FixedEffectOptimizationTracker(stateTracker))
  }

  def updateOffset(
    dataset: DataFrame, fixedEffectModel: FixedEffectModel, featureShardId: FeatureShardId,
    inputColumnsNames: InputColumnsNames): DataFrame = {

    require(
      featureShardId == fixedEffectModel.featureShardId,
      s"Fixed effect coordinate featureShardId ${featureShardId} != model.featureShardId ${
        fixedEffectModel
          .featureShardId
      }")

    val offset = inputColumnsNames(InputColumnsNames.OFFSET)
    val hasOffsetField = ApiUtils.hasColumn(dataset, offset)
    val hasCoordinateScoreField = ApiUtils.hasColumn(dataset, SCORE_FIELD)

    if (hasOffsetField && hasCoordinateScoreField) {
      // offset = offset - old_coordinateScore + new_coordinateScore
      dataset.withColumn(offset, col(offset) - col(SCORE_FIELD))
      fixedEffectModel.computeScore(dataset, SCORE_FIELD)
        .withColumn(offset, col(offset) + col(SCORE_FIELD))
    } else if (!hasOffsetField && !hasCoordinateScoreField) {
      fixedEffectModel.computeScore(dataset, SCORE_FIELD)
        .withColumn(offset, col(SCORE_FIELD))
    } else if (hasOffsetField && !hasCoordinateScoreField) {
      fixedEffectModel.computeScore(dataset, SCORE_FIELD)
        .withColumn(offset, col(offset) + col(SCORE_FIELD))
    } else {
      throw new UnsupportedOperationException("It shouldn't happen!")
    }
  }
}
