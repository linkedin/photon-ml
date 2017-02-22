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
package com.linkedin.photon.ml.algorithm

import com.linkedin.photon.ml.data.{KeyValueScore, RandomEffectDataSet, RandomEffectDataSetInProjectedSpace}
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.{Coefficients, DatumScoringModel, RandomEffectModel, RandomEffectModelInProjectedSpace}
import com.linkedin.photon.ml.optimization.OptimizationTracker
import com.linkedin.photon.ml.optimization.game.RandomEffectOptimizationProblem
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * The optimization problem coordinate for a random effect model in projected space.
 *
 * @tparam Objective The type of objective function used to solve individual random effect optimization problems
 * @param dataSetInProjectedSpace The training dataset
 * @param optimizationProblem The fixed effect optimization problem
 */
protected[ml] class RandomEffectCoordinateInProjectedSpace[Objective <: SingleNodeObjectiveFunction](
    dataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace,
    optimizationProblem: RandomEffectOptimizationProblem[Objective])
  extends RandomEffectCoordinate[Objective](dataSetInProjectedSpace, optimizationProblem) {

  /**
   * Score the effect-specific data set in the coordinate with the input model.
   *
   * @param model The input model
   * @return The output scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): KeyValueScore = model match {
    case randomEffectModelWithProjector: RandomEffectModelInProjectedSpace =>
      val randomEffectModel = randomEffectModelWithProjector.toRandomEffectModel
      super.score(randomEffectModel)

    case _ =>
      throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
        s"in ${this.getClass} is not supported!")
  }

  /**
   * Initialize a basic model for scoring GAME data.
   *
   * @param seed A random seed
   * @return The basic model
   */
  override protected[algorithm] def initializeModel(seed: Long): RandomEffectModelInProjectedSpace =
    RandomEffectCoordinateInProjectedSpace.initializeModel(dataSetInProjectedSpace, optimizationProblem)

  /**
   * Update the coordinate with a new dataset.
   *
   * @param updatedRandomEffectDataSet The updated dataset
   * @return A new coordinate with the updated dataset
   */
  override protected[algorithm] def updateCoordinateWithDataSet(
      updatedRandomEffectDataSet: RandomEffectDataSet): RandomEffectCoordinate[Objective] = {

    val updatedRandomEffectDataSetInProjectedSpace = new RandomEffectDataSetInProjectedSpace(
      updatedRandomEffectDataSet,
      dataSetInProjectedSpace.randomEffectProjector)

    new RandomEffectCoordinateInProjectedSpace(
      updatedRandomEffectDataSetInProjectedSpace,
      optimizationProblem)
  }

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
      case randomEffectModelWithProjector: RandomEffectModelInProjectedSpace =>
        val randomEffectModel = randomEffectModelWithProjector.toRandomEffectModel
        val (updatedModel, optimizationTracker) = super.updateModel(randomEffectModel)
        val updatedModelsRDD = updatedModel.asInstanceOf[RandomEffectModel].modelsRDD
        val updatedRandomEffectModelWithProjector = randomEffectModelWithProjector
          .updateRandomEffectModelInProjectedSpace(updatedModelsRDD)

        (updatedRandomEffectModelWithProjector, optimizationTracker)

      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} in ${this.getClass} is " +
            s"not supported!")
    }

  /**
   * Compute the regularization term value of the coordinate for a given model.
   *
   * @param model The model
   * @return The regularization term value
   */
  override protected[algorithm] def computeRegularizationTermValue(model: DatumScoringModel): Double = model match {
    case randomEffectModelWithProjector: RandomEffectModelInProjectedSpace =>
      val randomEffectModel = randomEffectModelWithProjector.toRandomEffectModel
      super.computeRegularizationTermValue(randomEffectModel)

    case _ =>
      throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
        s"type ${model.getClass} in ${this.getClass} is not supported!")
  }
}

object RandomEffectCoordinateInProjectedSpace {
  /**
   * Initialize a basic model (one that has a zero model for each random effect).
   *
   * @tparam Function The type of objective function used to solve individual random effect optimization problems
   * @param randomEffectDataSetInProjectedSpace The dataset
   * @param randomEffectOptimizationProblem The optimization problem to use for creating the underlying models
   * @return A random effect model for scoring GAME data
   */
  private def initializeModel[Function <: SingleNodeObjectiveFunction](
      randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace,
      randomEffectOptimizationProblem: RandomEffectOptimizationProblem[Function]): RandomEffectModelInProjectedSpace = {

    val glm = randomEffectOptimizationProblem.initializeModel(0)
    val randomEffectModelsRDD = randomEffectDataSetInProjectedSpace.activeData.mapValues { localDataSet =>
      glm.updateCoefficients(Coefficients.initializeZeroCoefficients(localDataSet.numFeatures))
        .asInstanceOf[GeneralizedLinearModel]
    }
    val randomEffectType = randomEffectDataSetInProjectedSpace.randomEffectType
    val featureShardId = randomEffectDataSetInProjectedSpace.featureShardId
    val randomEffectProjector = randomEffectDataSetInProjectedSpace.randomEffectProjector

    new RandomEffectModelInProjectedSpace(
      randomEffectModelsRDD,
      randomEffectProjector,
      randomEffectType,
      featureShardId)
  }
}
