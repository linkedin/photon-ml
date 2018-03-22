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
 * Performs a random search of the space whose bounds are specified by the given ranges
 *
 * @param ranges the ranges that define the boundaries of the search space
 * @param evaluationFunction the function that evaluates points in the space to real values
 * @param discreteParams specifies the indices of parameters that should be treated as discrete values
 * @param seed the random seed value
 */
class RandomSearch[T](
    ranges: Seq[DoubleRange],
    evaluationFunction: EvaluationFunction[T],
    discreteParams: Seq[Int] = Seq(),
    seed: Long = System.currentTimeMillis) {

  // The length of the ranges sequence corresponds to the dimensionality of the hyperparameter tuning problem
  protected val numParams: Int = ranges.length

  /**
   * Sobol generator for uniformly choosing rougly equidistant points
   */
  private val paramDistributions = {
    val sobol = new SobolSequenceGenerator(numParams)
    sobol.skipTo((seed % (Int.MaxValue.toLong + 1)).toInt)

    sobol
  }

  /**
   * Searches and returns n points in the space with the given prior observations
   *
   * @param n the number of points to find
   * @param observations observations made prior to searching, as (paramVector, evaluationValue) tuples in the current
   *                     dataset
   * @param priorObservations observations from the past datasets. These are consider as mean centered.
   * @return the found points
   */
  def findWithPrior(
      n: Int,
      observations: Seq[(DenseVector[Double], Double)],
      priorObservations: Option[Seq[(DenseVector[Double], Double)]] = None): Seq[T] = {
    require(n > 0, "The number of results must be greater than zero.")

    // Load the initial observations
    observations.init.foreach { case (candidate, value) =>
      onObservation(candidate, value)
    }

    if(priorObservations.isDefined) {
      // Load the prior observations. We add all of them since we do not iterate over these.
      priorObservations.get.foreach { case (candidate, value) =>
        onObservation(candidate, value, priorObservations.isDefined)
      }
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
   * Searches and returns n points in the space
   *
   * @param n the number of points to find
   * @param observations observations made prior to searching
   * @return the found points
   */
  def find(n: Int, observations: Seq[T]): Seq[T] = {
    require(n > 0, "The number of results must be greater than zero.")

    // Vectorize the initial observations
    val convertedObservations = observations.map { observation =>
      val candidate = evaluationFunction.vectorizeParams(observation)
      val value = evaluationFunction.getEvaluationValue(observation)
      (candidate, value)
    }

    findWithPrior(n, convertedObservations)
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

    // Discretize values specified as discrete
    discreteParams.foreach { index =>
      candidate(index) = round(candidate(index))
    }

    val (_, model) = evaluationFunction(candidate)

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
   * @param priorData the indicator to denote if this point and evaluation is coming from a past dataset
   */
  protected[search] def onObservation(point: DenseVector[Double], eval: Double, priorData: Boolean = false): Unit = {}

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
