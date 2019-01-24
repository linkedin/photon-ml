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

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.Types.{FeatureShardId, UniqueSampleId}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.function.DistributedObjectiveFunction
import com.linkedin.photon.ml.model.{DatumScoringModel, FixedEffectModel}
import com.linkedin.photon.ml.optimization.{DistributedOptimizationProblem, FixedEffectOptimizationTracker, OptimizationTracker}

/**
 * The optimization problem coordinate for a fixed effect model.
 *
 * @tparam Objective The type of objective function used to solve the fixed effect optimization problem
 * @param dataset The training dataset
 * @param optimizationProblem The fixed effect optimization problem
 */
protected[ml] class FixedEffectCoordinate[Objective <: DistributedObjectiveFunction](
    dataset: FixedEffectDataset,
    optimizationProblem: DistributedOptimizationProblem[Objective])
  extends Coordinate[FixedEffectDataset](dataset) {

  /**
   * Update the coordinate with a new dataset.
   *
   * @param dataset The updated dataset
   * @return A new coordinate with the updated dataset
   */
  override protected[algorithm] def updateCoordinateWithDataset(
      dataset: FixedEffectDataset): FixedEffectCoordinate[Objective] =
    new FixedEffectCoordinate[Objective](dataset, optimizationProblem)

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A tuple of the updated model and the optimization states tracker
   */
  override protected[algorithm] def trainModel(): (DatumScoringModel, Option[OptimizationTracker]) = {

    val updatedFixedEffectModel = FixedEffectCoordinate.trainModel(
      dataset.labeledPoints,
      optimizationProblem,
      dataset.featureShardId,
      None)
    val optimizationTracker = optimizationProblem.getStatesTracker.map(new FixedEffectOptimizationTracker(_))

    (updatedFixedEffectModel, optimizationTracker)
  }

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(
      model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker]) =
    model match {
      case fixedEffectModel: FixedEffectModel =>
        val updatedFixedEffectModel = FixedEffectCoordinate.trainModel(
          dataset.labeledPoints,
          optimizationProblem,
          dataset.featureShardId,
          Some(fixedEffectModel))
        val optimizationTracker = optimizationProblem.getStatesTracker.map(new FixedEffectOptimizationTracker(_))

        (updatedFixedEffectModel, optimizationTracker)

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
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = {
    model match {
      case fixedEffectModel: FixedEffectModel =>
        FixedEffectCoordinate.score(dataset, fixedEffectModel)

      case _ =>
        throw new UnsupportedOperationException(
          s"Scoring with model of type ${model.getClass} in ${this.getClass} is not supported")
    }
  }
}

object FixedEffectCoordinate {

  /**
   * Train a new [[FixedEffectModel]] (i.e. run model optimization).
   *
   * @param input The training dataset
   * @param optimizationProblem The optimization problem
   * @param featureShardId The ID of the feature shard for the training data
   * @param initialFixedEffectModelOpt An optional existing [[FixedEffectModel]] to use as a starting point for
   *                                   optimization
   * @return A new [[FixedEffectModel]]
   */
  private def trainModel[Function <: DistributedObjectiveFunction](
      input: RDD[(UniqueSampleId, LabeledPoint)],
      optimizationProblem: DistributedOptimizationProblem[Function],
      featureShardId: FeatureShardId,
      initialFixedEffectModelOpt: Option[FixedEffectModel]): FixedEffectModel = {

    val newModel = initialFixedEffectModelOpt
      .map { initialFixedEffectModel =>
        optimizationProblem.runWithSampling(input, initialFixedEffectModel.model)
      }
      .getOrElse(optimizationProblem.runWithSampling(input))
    val updatedModelBroadcast = input.sparkContext.broadcast(newModel)

    new FixedEffectModel(updatedModelBroadcast, featureShardId)
  }

  /**
   * Score a [[FixedEffectDataset]] using a given [[FixedEffectModel]].
   *
   * @note The score is the dot product of the model coefficients with the feature values (i.e., it does not go
   *       through a non-linear link function).
   * @param fixedEffectDataset The dataset to score
   * @param fixedEffectModel The model used to score the dataset
   * @return The computed scores
   */
  protected[algorithm] def score(
      fixedEffectDataset: FixedEffectDataset,
      fixedEffectModel: FixedEffectModel): CoordinateDataScores = {

    val modelBroadcast = fixedEffectModel.modelBroadcast
    val scores = fixedEffectDataset.labeledPoints.mapValues { case LabeledPoint(_, features, _, _) =>
      modelBroadcast.value.computeScore(features)
    }

    new CoordinateDataScores(scores)
  }
}
