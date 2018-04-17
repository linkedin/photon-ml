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

import scala.math.round

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.random.SobolSequenceGenerator

import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Performs a random search of the bounded space.
 *
 * @param ranges The ranges that define the boundaries of the search space
 * @param evaluationFunction The function that evaluates points in the space to real values
 * @param discreteParams Specifies the indices of parameters that should be treated as discrete values
 * @param seed A random seed
 */
class RandomSearch[T](
    ranges: Seq[DoubleRange],
    evaluationFunction: EvaluationFunction[T],
    discreteParams: Seq[Int] = Seq(),
    seed: Long = System.currentTimeMillis) {

  // The length of the ranges sequence corresponds to the dimensionality of the hyper-parameter tuning problem
  protected val numParams: Int = ranges.length

  /**
   * Sobol generator for uniformly choosing roughly equidistant points.
   */
  private val paramDistributions = {
    val sobol = new SobolSequenceGenerator(numParams)
    sobol.skipTo((seed % (Int.MaxValue.toLong + 1)).toInt)

    sobol
  }

  /**
   * Searches and returns n points in the space, given prior observations from this data set and past data sets.
   *
   * @param n The number of points to find
   * @param observations Observations made prior to searching, from this data set (not mean-centered)
   * @param priorObservations Observations made prior to searching, from past data sets (mean-centered)
   * @return The found points
   */
  def findWithPriors(
      n: Int,
      observations: Seq[(DenseVector[Double], Double)],
      priorObservations: Seq[(DenseVector[Double], Double)]): Seq[T] = {

    require(n > 0, "The number of results must be greater than zero.")

    // Load the initial observations
    observations.init.foreach { case (candidate, value) =>
      onObservation(candidate, value)
    }

    // Load the prior data observations
    priorObservations.foreach { case (candidate, value) =>
      onPriorObservation(candidate, value)
    }

    val (results, _) = (0 until n).foldLeft((List.empty[T], observations.last)) {
      case ((models, (lastCandidate, lastObservation)), _) =>

        val candidate = next(lastCandidate, lastObservation)

        // Discretize values specified as discrete
        discreteParams.foreach { index =>
          candidate(index) = round(candidate(index))
        }

        val (observation, model) = evaluationFunction(candidate)

        (models :+ model, (candidate, observation))
    }

    results
  }

  /**
   * Searches and returns n points in the space, given prior observations from this data set.
   *
   * @param n The number of points to find
   * @param observations Observations made prior to searching, from this data set (not mean-centered)
   * @return The found points
   */
  def findWithObservations(n: Int, observations: Seq[(DenseVector[Double], Double)]): Seq[T] =
    findWithPriors(n, observations, Seq())

  /**
   * Searches and returns n points in the space, given prior observations from past data sets.
   *
   * @param n The number of points to find
   * @param priorObservations Observations made prior to searching, from past data sets (mean-centered)
   * @return The found points
   */
  def findWithPriorObservations(n: Int, priorObservations: Seq[(DenseVector[Double], Double)]): Seq[T] = {

    require(n > 0, "The number of results must be greater than zero.")

    val candidate = drawCandidates(1)(0,::).t

    // Make values discrete as specified
    discreteParams.foreach { index =>
      candidate(index) = round(candidate(index))
    }

    val (_, model) = evaluationFunction(candidate)

    Seq(model) ++ (if (n == 1) Seq() else findWithPriors(n - 1, convertObservations(Seq(model)), priorObservations))
  }

  /**
   * Searches and returns n points in the space.
   *
   * @param n The number of points to find
   * @return The found points
   */
  def find(n: Int): Seq[T] = findWithPriorObservations(n, Seq())

  /**
   * Vectorize a [[Seq]] of prior observations.
   *
   * @param observations Prior observations in estimator output form
   * @return Prior observations as tuples of (vector representation of the original estimator output, evaluated value)
   */
  def convertObservations(observations: Seq[T]): Seq[(DenseVector[Double], Double)] =
    observations.map { observation =>
      val candidate = evaluationFunction.vectorizeParams(observation)
      val value = evaluationFunction.getEvaluationValue(observation)

      (candidate, value)
    }

  /**
   * Produces the next candidate, given the last. In this case, the next candidate is chosen uniformly from the space.
   *
   * @param lastCandidate the last candidate
   * @param lastObservation the last observed value
   * @return the next candidate
   */
  protected[search] def next(lastCandidate: DenseVector[Double], lastObservation: Double): DenseVector[Double] =
    drawCandidates(1)(0,::).t

  /**
   * Handler callback for each observation. In this case, we do nothing.
   *
   * @param point the observed point in the space
   * @param eval the observed value
   */
  protected[search] def onObservation(point: DenseVector[Double], eval: Double): Unit = {}

  /**
    * Handler callback for each observation in the prior data. In this case, we do nothing.
    *
    * @param point the observed point in the space
    * @param eval the observed value
    */
  protected[search] def onPriorObservation(point: DenseVector[Double], eval: Double): Unit = {}

  /**
   * Draw candidates from the distributions along each dimension in the space
   *
   * @param n the number of candidates to draw
   */
  protected[search] def drawCandidates(n: Int): DenseMatrix[Double] = {
    // Draw candidates from a Sobol generator, which produces values in the range [0, 1]
    val candidates = (1 until n).foldLeft(DenseMatrix(paramDistributions.nextVector)) { case (acc, _) =>
      DenseMatrix.vertcat(acc, DenseMatrix(paramDistributions.nextVector))
    }

    // Adjust candidates according to specified ranges
    ranges.zipWithIndex.foreach { case (range, j) =>
      candidates(::,j) *= (range.end - range.start)
      candidates(::,j) += range.start
    }

    candidates
  }
}
