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

import math.max

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sqrt
import breeze.stats.variance

import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.hyperparameter.criteria.ConfidenceBound
import com.linkedin.photon.ml.hyperparameter.estimators.{GaussianProcessEstimator, GaussianProcessModel}
import com.linkedin.photon.ml.hyperparameter.estimators.kernels.Matern52

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
 * @param candidatePoolSize the number of candidate points to draw at each iteration. Larger numbers give more precise
 *   results, but also incur higher computational cost.
 * @param seed the random seed value
 */
class GaussianProcessSearch[T](
    ranges: Seq[(Double, Double)],
    evaluationFunction: EvaluationFunction[T],
    evaluator: Evaluator,
    candidatePoolSize: Int = 250,
    seed: Long = System.currentTimeMillis)
  extends RandomSearch[T](ranges, evaluationFunction, seed){

  private var observedPoints: Option[DenseMatrix[Double]] = None
  private var observedEvals: Option[DenseVector[Double]] = None
  private var bestEval: Double = evaluator.defaultScore
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

        // The confidence bound criteria produced the best results on our synthesized data sets and real world
        // tests. Here we derive the exploration factor from the sample variance of previous observations.
        val obsStd = sqrt(max(1.0, variance(evals)))
        val transformation = new ConfidenceBound(evaluator, 2*obsStd)

        val estimator = new GaussianProcessEstimator(
          kernel = kernel,
          normalizeLabels = true,
          predictionTransformation = Some(transformation),
          seed = seed)

        val model = estimator.fit(points, evals)
        lastModel = model

        val predictions = model.predictTransformed(candidates)

        selectBestCandidate(candidates, predictions)

      // If we've received fewer observations than the number of parameters, fall back to a uniform search, to ensure
      // that the problem is not underdetermined.
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

    if (evaluator.betterThan(eval, bestEval)) {
      bestEval = eval
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
