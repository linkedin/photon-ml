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

import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.{FixedEffectDataSet, KeyValueScore, LabeledPoint}
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.model.{DatumScoringModel, FixedEffectModel}
import com.linkedin.photon.ml.normalization.NoNormalization
import com.linkedin.photon.ml.optimization.GeneralizedLinearOptimizationProblem
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationTracker, OptimizationTracker}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.PhotonLogger

/**
  * The optimization problem coordinate for a fixed effect model
  *
  * @param fixedEffectDataSet The training dataset
  * @param optimizationProblem The fixed effect optimization problem
  */
protected[ml] class FixedEffectCoordinate[GLM <: GeneralizedLinearModel, F <: DiffFunction[LabeledPoint]](
    fixedEffectDataSet: FixedEffectDataSet,
    private var optimizationProblem: GeneralizedLinearOptimizationProblem[GLM, F])
  extends Coordinate[FixedEffectDataSet, FixedEffectCoordinate[GLM, F]](fixedEffectDataSet) {

  /**
    * Initialize the model
    *
    * @param seed Random seed
    */
  protected[algorithm] def initializeModel(seed: Long): FixedEffectModel = {
    val numFeatures = fixedEffectDataSet.numFeatures
    val generalizedLinearModel = optimizationProblem.initializeZeroModel(numFeatures)
    val generalizedLinearModelBroadcast = fixedEffectDataSet.sparkContext.broadcast(
      generalizedLinearModel.asInstanceOf[GeneralizedLinearModel])
    val featureShardId = fixedEffectDataSet.featureShardId
    new FixedEffectModel(generalizedLinearModelBroadcast, featureShardId)
  }

  /**
    * Update the coordinate with a dataset
    *
    * @param fixedEffectDataSet The updated dataset
    * @return The updated coordinate
    */
  override protected def updateCoordinateWithDataSet(fixedEffectDataSet: FixedEffectDataSet)
    : FixedEffectCoordinate[GLM, F] = new FixedEffectCoordinate[GLM, F](fixedEffectDataSet, optimizationProblem)

  /**
    * Update the model
    *
    * @param model The model to update
    */
  protected[algorithm] override def updateModel(model: DatumScoringModel)
    : (DatumScoringModel, OptimizationTracker) = model match {

    case fixedEffectModel: FixedEffectModel =>
      val (updatedFixedEffectModel, updatedOptimizationProblem) = FixedEffectCoordinate.updateModel(
        fixedEffectDataSet,
        optimizationProblem,
        fixedEffectModel)
      //Note that the optimizationProblem will memorize the current state of optimization,
      //and the next round of updating global models will share the same convergence criteria as this one.
      optimizationProblem = updatedOptimizationProblem

      val optimizationTracker = new FixedEffectOptimizationTracker(optimizationProblem.getStatesTracker.get)

      (updatedFixedEffectModel, optimizationTracker)

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
  protected[algorithm] def score(model: DatumScoringModel): KeyValueScore = {
    model match {
      case fixedEffectModel: FixedEffectModel =>
        FixedEffectCoordinate.updateScore(fixedEffectDataSet, fixedEffectModel)

      case _ =>
        throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
            s"in ${this.getClass} is not supported!")
    }
  }

  /**
    * Compute the regularization term value
    *
    * @param model The model
    * @return Regularization term value
    */
  protected[algorithm] def computeRegularizationTermValue(model: DatumScoringModel): Double = {
    model match {
      case fixedEffectModel: FixedEffectModel =>
        optimizationProblem.getRegularizationTermValue(fixedEffectModel.model)

      case _ =>
        throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
            s"type ${model.getClass} in ${this.getClass} is not supported!")
    }
  }

  /**
    * Summarize the coordinate state
    *
    * @param logger A logger instance
    */
  protected[algorithm] def summarize(logger: PhotonLogger): Unit = {
    logger.debug(s"Optimization stats: ${optimizationProblem.getStatesTracker.get}")
  }
}

object FixedEffectCoordinate {
  /**
    * Update the model (i.e. run the coordinate optimizer)
    *
    * @param fixedEffectDataSet The dataset
    * @param optimizationProblem The optimization problem
    * @param fixedEffectModel The model
    * @return Tuple of updated model and optimization tracker
    */
  private def updateModel[GLM <: GeneralizedLinearModel, F <: DiffFunction[LabeledPoint]](
      fixedEffectDataSet: FixedEffectDataSet,
      optimizationProblem: GeneralizedLinearOptimizationProblem[GLM, F],
      fixedEffectModel: FixedEffectModel): (FixedEffectModel, GeneralizedLinearOptimizationProblem[GLM, F]) = {

    val trainingData = optimizationProblem
      .downSample(fixedEffectDataSet.labeledPoints)
      .setName("In memory fixed effect training data set")
      .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val model = fixedEffectModel.model
    // TODO: Allow normalization
    val updateModel = optimizationProblem.run(trainingData.values, model, NoNormalization)
    val updateModelBroadcast = fixedEffectDataSet
      .sparkContext
      .broadcast(updateModel.asInstanceOf[GeneralizedLinearModel])
    val updatedFixedEffectModel = fixedEffectModel.update(updateModelBroadcast)
    trainingData.unpersist()

    (updatedFixedEffectModel, optimizationProblem)
  }

  /**
    * Compute updated scores
    *
    * @param fixedEffectDataSet The dataset
    * @param fixedEffectModel The model
    * @return Scores
    */
  private def updateScore(fixedEffectDataSet: FixedEffectDataSet, fixedEffectModel: FixedEffectModel): KeyValueScore = {
    val modelBroadcast = fixedEffectModel.modelBroadcast
    val scores = fixedEffectDataSet.labeledPoints.mapValues { case LabeledPoint(_, features, _, _) =>
      modelBroadcast.value.computeScore(features)
    }

    new KeyValueScore(scores)
  }
}
