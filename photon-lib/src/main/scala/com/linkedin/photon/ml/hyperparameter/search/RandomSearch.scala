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

import scala.math.floor
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.random.SobolSequenceGenerator
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.hyperparameter.estimators.kernels.{Matern52, StationaryKernel}


/**
 * Performs a random search of the bounded space.
 *
 * @param numParams The dimensionality of the hyper-parameter tuning problem
 * @param evaluationFunction The function that evaluates points in the space to real values
 * @param discreteParams Specifies the indices of discrete parameters and their numbers of discrete values
 * @param kernel Specifies the indices and transformation function of hyper-parameters
 * @param seed A random seed
 */
class RandomSearch[T](
    numParams: Int,
    evaluationFunction: EvaluationFunction[T],
    discreteParams: Map[Int, Int] = Map(),
    kernel: StationaryKernel = new Matern52,
    seed: Long = System.currentTimeMillis) {

  require(numParams > 0, "Number of parameters must be non-negative.")

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
    require(observations.nonEmpty, "There must be at least one observation.")

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
        val candidateWithDiscrete = discretizeCandidate(candidate, discreteParams)

        val (observation, model) = evaluationFunction(candidateWithDiscrete)

        (models :+ model, (candidateWithDiscrete, observation))
    }

    results
  }

  /**
   * Searches and returns n points in the space, given prior observations from past data sets.
   *
   * @param n The number of points to find
   * @param priorObservations Observations made prior to searching, from past data sets (mean-centered)
   * @return The found points
   */
  def findWithPriorObservations(n: Int, priorObservations: Seq[(DenseVector[Double], Double)]): Seq[T] = {

    require(n > 0, "The number of results must be greater than zero.")

    val candidate = drawCandidates(1)(0, ::).t

    // Make values discrete as specified
    val candidateWithDiscrete = discretizeCandidate(candidate, discreteParams)

    val (_, model) = evaluationFunction(candidateWithDiscrete)
    val initialObservation = evaluationFunction.convertObservations(Seq(model))

    Seq(model) ++ (if (n == 1) Seq() else findWithPriors(n - 1, initialObservation, priorObservations))
  }


  /**
   * Searches and returns n points in the space.
   *
   * @param n The number of points to find
   * @return The found points
   */
  def find(n: Int): Seq[T] = findWithPriorObservations(n, Seq())

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
    (1 until n).foldLeft(DenseMatrix(paramDistributions.nextVector)) { case (acc, _) =>
      DenseMatrix.vertcat(acc, DenseMatrix(paramDistributions.nextVector))
    }
  }

  /**
   * Discretize candidates with specified indices.
   *
   * @param candidate candidate with values in [0, 1]
   * @param discreteParams Map that specifies the indices of discrete parameters and their numbers of discrete values
   * @return candidate with the specified discrete values
   */
  protected[search] def discretizeCandidate(
      candidate: DenseVector[Double],
      discreteParams: Map[Int, Int]): DenseVector[Double] = {

    val candidateWithDiscrete = candidate.copy

    discreteParams.foreach { case (index, numDiscreteValues) =>
      candidateWithDiscrete(index) = floor(candidate(index) * numDiscreteValues) / numDiscreteValues
    }

    candidateWithDiscrete
  }
}
