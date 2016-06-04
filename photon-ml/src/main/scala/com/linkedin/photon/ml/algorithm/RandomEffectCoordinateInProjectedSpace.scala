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

import com.linkedin.photon.ml.data.{KeyValueScore, LabeledPoint, RandomEffectDataSet, RandomEffectDataSetInProjectedSpace}
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.model.{DatumScoringModel, RandomEffectModel, RandomEffectModelInProjectedSpace}
import com.linkedin.photon.ml.optimization.game.{OptimizationTracker, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
  * The optimization problem coordinate for a random effect model in projected space
  *
  * @param randomEffectDataSetInProjectedSpace The training dataset
  * @param randomEffectOptimizationProblem The fixed effect optimization problem
  */
protected[ml] class RandomEffectCoordinateInProjectedSpace[GLM <: GeneralizedLinearModel, F <: DiffFunction[LabeledPoint]](
    randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace,
    randomEffectOptimizationProblem: RandomEffectOptimizationProblem[GLM, F])
  extends RandomEffectCoordinate[GLM, F](randomEffectDataSetInProjectedSpace, randomEffectOptimizationProblem) {

  /**
    * Initialize the model
    *
    * @param seed Random seed
    */
  protected[algorithm] override def initializeModel(seed: Long): RandomEffectModelInProjectedSpace = {
    RandomEffectCoordinateInProjectedSpace.initializeModel(
      randomEffectDataSetInProjectedSpace,
      randomEffectOptimizationProblem)
  }

  /**
    * Update the model (i.e. run the coordinate optimizer)
    *
    * @param model The model
    * @return Tuple of updated model and optimization tracker
    */
  protected[algorithm] override def updateModel(model: DatumScoringModel): (DatumScoringModel, OptimizationTracker) =
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
    * Score the model
    *
    * @param model The model to score
    * @return Scores
    */
  protected[algorithm] override def score(model: DatumScoringModel): KeyValueScore = model match {
    case randomEffectModelWithProjector: RandomEffectModelInProjectedSpace =>
      val randomEffectModel = randomEffectModelWithProjector.toRandomEffectModel
      super.score(randomEffectModel)

    case _ =>
      throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
        s"in ${this.getClass} is not supported!")
  }

  /**
    * Compute the regularization term value
    *
    * @param model The model
    * @return Regularization term value
    */
  protected[algorithm] override def computeRegularizationTermValue(model: DatumScoringModel): Double = model match {
    case randomEffectModelWithProjector: RandomEffectModelInProjectedSpace =>
      val randomEffectModel = randomEffectModelWithProjector.toRandomEffectModel
      super.computeRegularizationTermValue(randomEffectModel)

    case _ =>
      throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
        s"type ${model.getClass} in ${this.getClass} is not supported!")
  }

  /**
    * Update the coordinate with a dataset
    *
    * @param updatedRandomEffectDataSet The updated dataset
    * @return The updated coordinate
    */
  override protected def updateCoordinateWithDataSet(updatedRandomEffectDataSet: RandomEffectDataSet)
    : RandomEffectCoordinate[GLM, F] = {

    val updatedRandomEffectDataSetInProjectedSpace = new RandomEffectDataSetInProjectedSpace(
      updatedRandomEffectDataSet,
      randomEffectDataSetInProjectedSpace.randomEffectProjector)

    new RandomEffectCoordinateInProjectedSpace(
      updatedRandomEffectDataSetInProjectedSpace,
      randomEffectOptimizationProblem)
  }
}

object RandomEffectCoordinateInProjectedSpace {

  /**
    * Initialize a zero model
    *
    * @param randomEffectDataSetInProjectedSpace The dataset
    */
  private def initializeModel[GLM <: GeneralizedLinearModel, F <: DiffFunction[LabeledPoint]](
      randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace,
      randomEffectOptimizationProblem: RandomEffectOptimizationProblem[GLM, F]): RandomEffectModelInProjectedSpace = {

    val randomEffectModelsRDD = randomEffectDataSetInProjectedSpace.activeData.mapValues { localDataSet =>
      randomEffectOptimizationProblem.initializeModel(localDataSet.numFeatures).asInstanceOf[GeneralizedLinearModel]
    }
    val randomEffectId = randomEffectDataSetInProjectedSpace.randomEffectId
    val featureShardId = randomEffectDataSetInProjectedSpace.featureShardId
    val randomEffectProjector = randomEffectDataSetInProjectedSpace.randomEffectProjector

    new RandomEffectModelInProjectedSpace(randomEffectModelsRDD, randomEffectProjector, randomEffectId, featureShardId)
  }
}
