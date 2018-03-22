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

import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.hyperparameter.criteria.ExpectedImprovement
import com.linkedin.photon.ml.hyperparameter.estimators.{GaussianProcessEstimator, GaussianProcessModel}
import com.linkedin.photon.ml.hyperparameter.estimators.kernels.Matern52
import com.linkedin.photon.ml.util.DoubleRange

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
 * @param ranges the value ranges within which to search. There should be one for every dimension in the space.
 * @param evaluationFunction the function that evaluates points in the space to real values
 * @param evaluator the original evaluator
 * @param discreteParams specifies the indices of parameters that should be treated as discrete values
 * @param candidatePoolSize the number of candidate points to draw at each iteration. Larger numbers give more precise
 *   results, but also incur higher computational cost.
 * @param noisyTarget whether to include observation noise in the evaluation function model
 * @param seed the random seed value
 */
class GaussianProcessSearch[T](
    ranges: Seq[DoubleRange],
    evaluationFunction: EvaluationFunction[T],
    evaluator: Evaluator,
    discreteParams: Seq[Int] = Seq(),
    candidatePoolSize: Int = 250,
    noisyTarget: Boolean = false,
    seed: Long = System.currentTimeMillis)
  extends RandomSearch[T](ranges, evaluationFunction, discreteParams, seed){

  private var observedPoints: Option[DenseMatrix[Double]] = None
  private var observedEvals: Option[DenseVector[Double]] = None
  private var bestEval: Double = evaluator.defaultScore
  private var priorObservedPoints: Option[DenseMatrix[Double]] = None
  private var priorObservedEvals: Option[DenseVector[Double]] = None
  private var priorBestEval: Double = evaluator.defaultScore
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

        // We choose the Matérn 5/2 covariance kernel since it performs best in the literature and our tests for
        // hyperparameter spaces.
        val kernel = new Matern52

        //Finding the overall bestEval
        val currentMean =  mean(evals)
        val overallBestEval = if(evaluator.betterThan(priorBestEval, bestEval - currentMean)) {
          priorBestEval
        } else {
          bestEval - currentMean
        }

        // Expected improvement transformation
        val transformation = new ExpectedImprovement(evaluator, overallBestEval)

        val estimator = new GaussianProcessEstimator(
          kernel = kernel,
          normalizeLabels = false,
          noisyTarget = noisyTarget,
          predictionTransformation = Some(transformation),
          seed = seed)

        //Union of points and evals with priorData
        val (overallPoints, overallEvals) = if(priorObservedPoints.isDefined) {
          (DenseMatrix.vertcat(points, priorObservedPoints.get),
            DenseVector.vertcat(evals - currentMean, priorObservedEvals.get))
        } else {
          (points, evals - currentMean)
        }

        val model = estimator.fit(overallPoints, overallEvals)
        lastModel = model

        val predictions = model.predictTransformed(candidates)

        selectBestCandidate(candidates, predictions)

      // If we've received fewer observations than the number of parameters, fall back to a uniform search, to ensure
      // that the problem is not underdetermined.
      case _ => super.next(lastCandidate, lastObservation)
    }
  }

  /**
   * Handler for adding new points and evaluation values
   *
   * @param pastPoints the past set of points
   * @param pastEvals the past current set of evaluation values
   * @param pastBestEval the past best value
   * @param point the new point to be added
   * @param eval the new evaluation value
   *
   * @return the new set of points, evaluations and best evaluation
   */
  protected[search] def addObservation(
      pastPoints: Option[DenseMatrix[Double]],
      pastEvals: Option[DenseVector[Double]],
      pastBestEval: Double,
      point: DenseVector[Double],
      eval: Double): (Option[DenseMatrix[Double]], Option[DenseVector[Double]], Double) = {

    val newPoints = pastPoints
      .map(DenseMatrix.vertcat(_, point.toDenseMatrix))
      .orElse(Some(point.toDenseMatrix))

    val newEvals = pastEvals
      .map(DenseVector.vertcat(_, DenseVector(eval)))
      .orElse(Some(DenseVector(eval)))

    val newBest = if (evaluator.betterThan(eval, pastBestEval)) { eval } else { pastBestEval }

    (newPoints, newEvals, newBest)
  }

  /**
   * Handler callback for each observation. In this case, we record the observed point and values.
   *
   * @param point the observed point in the space
   * @param eval the observed value
   */
  protected[search] override def onObservation(
      point: DenseVector[Double],
      eval: Double,
      priorData: Boolean = false): Unit = {

    if(priorData) {
      val (newPoints, newEvals, newBest) = addObservation(priorObservedPoints,
        priorObservedEvals,
        priorBestEval,
        point,
        eval)
      priorObservedPoints = newPoints
      priorObservedEvals = newEvals
      priorBestEval = newBest
    } else {
      val (newPoints, newEvals, newBest) = addObservation(observedPoints,
        observedEvals,
        bestEval,
        point,
        eval)
      observedPoints = newPoints
      observedEvals = newEvals
      bestEval = newBest
    }
  }

  /**
   * Selects the best candidate according to the predicted values, where "best" is determined by the given evaluator
   *
   * @param candidates matrix of candidates
   * @param predictions predicted values for each candidate
   * @return the candidate with the best value
   */
  protected[search] def selectBestCandidate(
      candidates: DenseMatrix[Double],
      predictions: DenseVector[Double]): DenseVector[Double] = {

    val init = (candidates(0,::).t, predictions(0))

    val (selectedCandidate, _) = (1 until candidates.rows).foldLeft(init) {
      case ((bestCandidate, bestPrediction), i) =>
        if (evaluator.betterThan(predictions(i), bestPrediction)) {
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
