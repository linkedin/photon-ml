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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.function.DistributedObjectiveFunction
import com.linkedin.photon.ml.model.{DatumScoringModel, FixedEffectModel}
import com.linkedin.photon.ml.optimization.{DistributedOptimizationProblem, FixedEffectOptimizationTracker, OptimizationTracker}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * The optimization problem coordinate for a fixed effect model.
 *
 * @tparam Objective The type of objective function used to solve the fixed effect optimization problem
 * @param dataSet The training dataset
 * @param optimizationProblem The fixed effect optimization problem
 */
protected[ml] class FixedEffectCoordinate[Objective <: DistributedObjectiveFunction](
    dataSet: FixedEffectDataSet,
    optimizationProblem: DistributedOptimizationProblem[Objective])
  extends Coordinate[FixedEffectDataSet](dataSet) {

  /**
   * Score the effect-specific data set in the coordinate with the input model.
   *
   * @param model The input model
   * @return The output scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = {
    model match {
      case fixedEffectModel: FixedEffectModel =>
        FixedEffectCoordinate.score(dataSet, fixedEffectModel)

      case _ =>
        throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
          s"in ${this.getClass} is not supported")
    }
  }

  /**
   * Initialize a basic model for scoring GAME data.
   *
   * @param seed A random seed
   * @return The basic model
   */
  override protected[algorithm] def initializeModel(seed: Long): FixedEffectModel = {
    val numFeatures = dataSet.numFeatures
    val generalizedLinearModel = optimizationProblem.initializeZeroModel(numFeatures)
    val generalizedLinearModelBroadcast = dataSet
      .sparkContext
      .broadcast(generalizedLinearModel.asInstanceOf[GeneralizedLinearModel])
    val featureShardId = dataSet.featureShardId

    new FixedEffectModel(generalizedLinearModelBroadcast, featureShardId)
  }

  /**
   * Update a coordinate with a new dataset.
   *
   * @param dataSet The updated dataset
   * @return A new coordinate with the updated dataset
   */
  override protected[algorithm] def updateCoordinateWithDataSet(
      dataSet: FixedEffectDataSet): FixedEffectCoordinate[Objective] =
    new FixedEffectCoordinate[Objective](dataSet, optimizationProblem)

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A tuple of the updated model and the optimization states tracker
   */
  override protected[algorithm] def updateModel(
      model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker]) =
    model match {
      case fixedEffectModel: FixedEffectModel =>
        val updatedFixedEffectModel = FixedEffectCoordinate.updateModel(
          dataSet.labeledPoints,
          optimizationProblem,
          fixedEffectModel,
          dataSet.sparkContext)
        val optimizationTracker = optimizationProblem.getStatesTracker.map(new FixedEffectOptimizationTracker(_))

        (updatedFixedEffectModel, optimizationTracker)

      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} in ${this.getClass} is " +
            s"not supported")
    }

  /**
   * Compute the regularization term value of the coordinate for a given model.
   *
   * @param model The model
   * @return The regularization term value
   */
  override protected[algorithm] def computeRegularizationTermValue(model: DatumScoringModel): Double =
    model match {
      case fixedEffectModel: FixedEffectModel =>
        optimizationProblem.getRegularizationTermValue(fixedEffectModel.model)

      case _ =>
        throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
            s"type ${model.getClass} in ${this.getClass} is not supported")
    }
}

object FixedEffectCoordinate {
  /**
   * Update the model (i.e. run the coordinate optimizer).
   *
   * @param input The training dataset
   * @param optimizationProblem The optimization problem
   * @param fixedEffectModel The current model, used as a starting point
   * @param sc The current Spark context
   * @return A tuple of the optimized model and the updated optimization problem
   */
  private def updateModel[Function <: DistributedObjectiveFunction](
      input: RDD[(UniqueSampleId, LabeledPoint)],
      optimizationProblem: DistributedOptimizationProblem[Function],
      fixedEffectModel: FixedEffectModel,
      sc: SparkContext): FixedEffectModel = {

    val model = fixedEffectModel.model
    val updatedModelBroadcast = sc.broadcast(optimizationProblem.runWithSampling(input, model))
    val updatedFixedEffectModel = fixedEffectModel.update(updatedModelBroadcast)

    updatedFixedEffectModel
  }

  /**
   * Score a dataset using a given model.
   *
   * @note The score is the dot product of the model coefficients with the feature values (in particular, does not go
   *       through non-linear link function in logistic regression!).
   * @param fixedEffectDataSet The dataset to score
   * @param fixedEffectModel The model to score the dataset with
   * @return The computed scores
   */
  protected[algorithm] def score(fixedEffectDataSet: FixedEffectDataSet, fixedEffectModel: FixedEffectModel): CoordinateDataScores = {
    val modelBroadcast = fixedEffectModel.modelBroadcast
    val scores = fixedEffectDataSet.labeledPoints.mapValues { case LabeledPoint(_, features, _, _) =>
      modelBroadcast.value.computeScore(features)
    }

    new CoordinateDataScores(scores)
  }
}
