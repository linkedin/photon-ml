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
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.{Coefficients, DatumScoringModel, FixedEffectModel}
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationTracker, OptimizationProblem, OptimizationTracker}
import com.linkedin.photon.ml.util.PhotonLogger

/**
  * The optimization problem coordinate for a fixed effect model
  *
  * @param fixedEffectDataSet The training dataset
  * @param optimizationProblem The fixed effect optimization problem
  */
protected[ml] class FixedEffectCoordinate[F <: TwiceDiffFunction[LabeledPoint]](
    fixedEffectDataSet: FixedEffectDataSet,
    private var optimizationProblem: OptimizationProblem[F])
  extends Coordinate[FixedEffectDataSet, FixedEffectCoordinate[F]](fixedEffectDataSet) {

  /**
    * Initialize the model
    *
    * @param seed Random seed
    */
  protected[algorithm] def initializeModel(seed: Long): FixedEffectModel = {
    FixedEffectCoordinate.initializeZeroModel(fixedEffectDataSet)
  }

  /**
    * Update the coordinate with a dataset
    *
    * @param fixedEffectDataSet The updated dataset
    * @return The updated coordinate
    */
  override protected def updateCoordinateWithDataSet(fixedEffectDataSet: FixedEffectDataSet)
    : FixedEffectCoordinate[F] = new FixedEffectCoordinate[F](fixedEffectDataSet, optimizationProblem)

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

        val optimizationTracker = new FixedEffectOptimizationTracker(optimizationProblem.optimizer.getStateTracker.get)

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
        optimizationProblem.getRegularizationTermValue(fixedEffectModel.coefficients)
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
    logger.debug(s"Optimization stats: ${optimizationProblem.optimizer.getStateTracker.get}")
  }
}

object FixedEffectCoordinate {

  /**
    * Initialize a zero model
    *
    * @param fixedEffectDataSet The dataset
    */
  private def initializeZeroModel(fixedEffectDataSet: FixedEffectDataSet): FixedEffectModel = {
    val numFeatures = fixedEffectDataSet.numFeatures
    val coefficients = Coefficients.initializeZeroCoefficients(numFeatures)
    val coefficientsBroadcast = fixedEffectDataSet.sparkContext.broadcast(coefficients)
    val featureShardId = fixedEffectDataSet.featureShardId
    new FixedEffectModel(coefficientsBroadcast, featureShardId)
  }

  /**
    * Update the model (i.e. run the coordinate optimizer)
    *
    * @param fixedEffectDataSet The dataset
    * @param optimizationProblem The optimization problem
    * @param fixedEffectModel The model
    * @return Tuple of updated model and optimization tracker
    */
  private def updateModel[F <: TwiceDiffFunction[LabeledPoint]](
      fixedEffectDataSet: FixedEffectDataSet,
      optimizationProblem: OptimizationProblem[F],
      fixedEffectModel: FixedEffectModel): (FixedEffectModel, OptimizationProblem[F]) = {

    val sampler = optimizationProblem.sampler
    val trainingData = sampler.downSample(fixedEffectDataSet.labeledPoints)
        .setName("In memory fixed effect training data set")
        .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val coefficients = fixedEffectModel.coefficients
    val (updatedCoefficients, _) = optimizationProblem.updateCoefficientMeans(trainingData.values, coefficients)
    val updatedCoefficientsBroadcast = fixedEffectDataSet.sparkContext.broadcast(updatedCoefficients)
    val updatedFixedEffectModel = fixedEffectModel.update(updatedCoefficientsBroadcast)
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
    val coefficientsBroadcast = fixedEffectModel.coefficientsBroadcast
    val scores = fixedEffectDataSet.labeledPoints.mapValues { case LabeledPoint(_, features, _, _) =>
      coefficientsBroadcast.value.computeScore(features)
    }

    new KeyValueScore(scores)
  }
}
