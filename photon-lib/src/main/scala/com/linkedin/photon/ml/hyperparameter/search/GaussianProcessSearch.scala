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
package com.linkedin.photon.ml.hyperparameter.search

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.mean

import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.hyperparameter.criteria.ExpectedImprovement
import com.linkedin.photon.ml.hyperparameter.estimators.{GaussianProcessEstimator, GaussianProcessModel, PredictionTransformation}
import com.linkedin.photon.ml.hyperparameter.estimators.kernels.{Matern52, StationaryKernel}

/**
 * Performs a guided random search of the given ranges, where the search is guided by a Gaussian Process estimated from
 * evaluations of the actual evaluation function. Since we assume that the evaluation function is very costly (as it
 * often is for doing a full train / cycle evaluation of a machine learning model), it makes sense to spend time doing
 * what would otherwise be considered an expensive computation to reduce the number of times we need to evaluate the
 * function.
 *
 * At a high level, the search routine proceeds as follows:
 *
 *  1) Assume a uniform prior over the evaluation function
 *  2) Receive a new observation, and use it along with any previous observations to train a new Gaussian Process
 *     regression model for the evaluation function. This approximation is the new posterior over the evaluation
 *     function.
 *  3) Sample candidates uniformly, evaluate the posterior for each, and select the candidate with the highest predicted
 *     evaluation.
 *  4) Evaluate the best candidate with the actual evaluation function to acquire a new observation.
 *  5) Repeat from step 2.
 *
 * @param numParams the dimensionality of the hyper-parameter tuning problem
 * @param evaluationFunction the function that evaluates points in the space to real values
 * @param discreteParams specifies the indices of discrete parameters and their numbers of discrete values
 * @param kernel specifies the covariance kernel for hyper-parameters
 * @param candidatePoolSize the number of candidate points to draw at each iteration. Larger numbers give more precise
 *   results, but also incur higher computational cost.
 * @param noisyTarget whether to include observation noise in the evaluation function model
 * @param seed the random seed value
 */
class GaussianProcessSearch[T](
    numParams: Int,
    evaluationFunction: EvaluationFunction[T],
    discreteParams: Map[Int, Int] = Map(),
    kernel: StationaryKernel = new Matern52,
    candidatePoolSize: Int = 250,
    noisyTarget: Boolean = true,
    seed: Long = System.currentTimeMillis)
  extends RandomSearch[T](numParams, evaluationFunction, discreteParams, kernel, seed){

  private var observedPoints: Option[DenseMatrix[Double]] = None
  private var observedEvals: Option[DenseVector[Double]] = None
  private var bestEval: Double = Double.PositiveInfinity
  private var priorObservedPoints: Option[DenseMatrix[Double]] = None
  private var priorObservedEvals: Option[DenseVector[Double]] = None
  private var priorBestEval: Double = Double.PositiveInfinity
  private var lastModel: GaussianProcessModel = _

  /**
   * Produces the next candidate, given the last. In this case, we fit a Gaussian Process to the previous observations,
   * and use it to predict the value of uniformly-drawn candidate points. The candidate with the best predicted
   * evaluation is chosen.
   *
   * @param lastCandidate the last candidate
   * @param lastObservation the last observed value
   * @return the next candidate
   */
  protected[search] override def next(
      lastCandidate: DenseVector[Double],
      lastObservation: Double): DenseVector[Double] = {

    onObservation(lastCandidate, lastObservation)

    (observedPoints, observedEvals) match {
      case (Some(points), Some(evals)) if points.rows > numParams =>
        val candidates = drawCandidates(candidatePoolSize)

        // Finding the overall bestEval
        val currentMean = mean(evals)
        val overallBestEval = Math.min(priorBestEval, bestEval - currentMean)

        // Expected improvement transformation
        val transformation = new ExpectedImprovement(overallBestEval)

        val estimator = new GaussianProcessEstimator(
          kernel = kernel,
          normalizeLabels = false,
          noisyTarget = noisyTarget,
          predictionTransformation = Some(transformation),
          seed = seed)

        // Union of points and evals with priorData
        val (overallPoints, overallEvals) = (priorObservedPoints, priorObservedEvals) match {
          case (Some(priorPoints), Some(priorEvals)) => (
            DenseMatrix.vertcat(points, priorPoints),
            DenseVector.vertcat(evals - currentMean, priorEvals))
          case _ =>
            (points, evals - currentMean)
        }

        val model = estimator.fit(overallPoints, overallEvals)
        lastModel = model

        val predictions = model.predictTransformed(candidates)

        selectBestCandidate(candidates, predictions, transformation)

      // If we've received fewer observations than the number of parameters, fall back to a uniform search, to ensure
      // that the problem is not under-determined.
      case _ => super.next(lastCandidate, lastObservation)
    }
  }

  /**
   * Handler callback for each observation. In this case, we record the observed point and values.
   *
   * @param point the observed point in the space
   * @param eval the observed value
   */
  protected[search] override def onObservation(point: DenseVector[Double], eval: Double): Unit = {
    observedPoints = observedPoints
      .map(DenseMatrix.vertcat(_, point.toDenseMatrix))
      .orElse(Some(point.toDenseMatrix))

    observedEvals = observedEvals
      .map(DenseVector.vertcat(_, DenseVector(eval)))
      .orElse(Some(DenseVector(eval)))

    bestEval = Math.min(bestEval, eval)
  }

  /**
    * Handler callback for each observation in the prior data. In this case, we record the observed point and values.
    *
    * @param point the observed point in the space
    * @param eval the observed value
    */
  protected[search] override def onPriorObservation(point: DenseVector[Double], eval: Double): Unit = {
    priorObservedPoints = priorObservedPoints
      .map(DenseMatrix.vertcat(_, point.toDenseMatrix))
      .orElse(Some(point.toDenseMatrix))

    priorObservedEvals = priorObservedEvals
      .map(DenseVector.vertcat(_, DenseVector(eval)))
      .orElse(Some(DenseVector(eval)))

    priorBestEval = Math.min(priorBestEval, eval)
  }

  /**
   * Selects the best candidate according to the predicted values, where "best" is determined by the given
   * transformation. In the case of EI, we always search for the max; In the case of CB, we always search for the min.
   *
   * @param candidates matrix of candidates
   * @param predictions predicted values for each candidate
   * @param transformation prediction transformation function
   * @return the candidate with the best value
   */
  protected[search] def selectBestCandidate(
      candidates: DenseMatrix[Double],
      predictions: DenseVector[Double],
      transformation: PredictionTransformation): DenseVector[Double] = {

    val init = (candidates(0,::).t, predictions(0))

    val direction = if (transformation.isMaxOpt) 1 else -1

    val (selectedCandidate, _) = (1 until candidates.rows).foldLeft(init) {
      case ((bestCandidate, bestPrediction), i) =>
        if (predictions(i) * direction > bestPrediction * direction) {
          (candidates(i,::).t, predictions(i))
        } else {
          (bestCandidate, bestPrediction)
        }
    }

    selectedCandidate
  }

  /**
   * Returns the last model trained during search
   *
   * @return the last model
   */
  def getLastModel: GaussianProcessModel = lastModel
}
