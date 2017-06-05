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
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator, Uniform}
import org.apache.commons.math3.random.MersenneTwister

import com.linkedin.photon.ml.hyperparameter.EvaluationFunction

/**
 * Performs a random search of the space whose bounds are specified by the given ranges
 *
 * @param ranges the ranges that define the boundaries of the search space
 * @param evaluationFunction the function that evaluates points in the space to real values
 * @param seed the random seed value
 */
class RandomSearch[T](
    ranges: Seq[(Double, Double)],
    evaluationFunction: EvaluationFunction[T],
    seed: Long = System.currentTimeMillis) {

  // The length of the ranges sequence corresponds to the dimensionality of the hyperparameter tuning problem
  protected val numParams = ranges.length

  /**
   * Provides an implicit random number basis for breeze that incorporates the given random seed.
   */
  private implicit val randBasis: RandBasis = new RandBasis(
    new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

  /**
   * Uniform distributions over parameters using the given ranges
   */
  private val paramDistributions = ranges.map { case (lower, upper) =>
    new Uniform(lower, upper)
  }

  /**
   * Searches and returns n points in the space
   *
   * @param n the number of points to find
   * @param observations observations made prior to searching
   * @return the found points
   */
  def find(n: Int, observations: Seq[T]): Seq[T] = {
    require(n > 0, "The number of results must be greater than zero.")

    // Vectorize and load the initial observations
    val convertedObservations = observations.map { observation =>
      val candidate = evaluationFunction.vectorizeParams(observation)
      val value = evaluationFunction.getEvaluationValue(observation)
      (candidate, value)
    }

    convertedObservations.init.foreach { case (candidate, value) =>
      onObservation(candidate, value)
    }

    val (results, _) = (0 until n).foldLeft((List.empty[T], convertedObservations.last)) {
      case ((models, (lastCandidate, lastObservation)), _) =>

      val candidate = next(lastCandidate, lastObservation)
      val (observation, model) = evaluationFunction(candidate)

      (models :+ model, (candidate, observation))
    }

    results
  }

  /**
   * Searches and returns n points in the space
   *
   * @param n the number of points to find
   * @return the found points
   */
  def find(n: Int): Seq[T] = {
    require(n > 0, "The number of results must be greater than zero.")

    val candidate = drawCandidates(1)(0,::).t
    val (observation, model) = evaluationFunction(candidate)

    Seq(model) ++ (if (n == 1) Seq() else find(n - 1, Seq(model)))
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
   * Draw candidates from the distributions along each dimension in the space
   *
   * @param n the number of candidates to draw
   */
  protected[search] def drawCandidates(n: Int): DenseMatrix[Double] =
    DenseMatrix.tabulate(n, numParams) { case (_, j) =>
      paramDistributions(j).draw
    }
}
