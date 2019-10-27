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

/*
 * @note This code is heavily influenced by the SPARK LIBLINEAR TRON implementation,
 * though not an exact copy. It also subject to the LIBLINEAR project's license
 * and copyright notice:
 *
 * Copyright (c) 2007-2015 The LIBLINEAR Project.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither name of copyright holders nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.linkedin.photon.ml.optimization

import breeze.linalg.{Vector, norm}

import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.{BroadcastWrapper, Logging, VectorUtils}

/**
 * This class used to solve an optimization problem using trust region Newton method (TRON).
 * Reference 1: [[http://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf]]
 * Reference 2:
 *   [[http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/distributed-liblinear/spark/running_spark_liblinear.html]]
 *
 * @param normalizationContext The normalization context
 * @param maxNumImprovementFailures The maximum allowed number of times in a row the objective hasn't improved. For
 *                                  most optimizers using line search, like L-BFGS, the improvement failure is not
 *                                  supposed to happen, because any improvement failure should be captured during the
 *                                  line search step. Here we are trying to capture the improvement failure after the
 *                                  gradient step. As a result, any improvement failure in this case results from some
 *                                  bug and we should not tolerate it. However, for optimizers like TRON occasional
 *                                  improvement failure is acceptable.
 * @param tolerance The tolerance threshold for improvement between iterations as a percentage of the initial loss
 * @param maxNumIterations The cut-off for number of optimization iterations to perform.
 * @param constraintMap (Optional) The map of constraints on the feature coefficients
 */
class TRON(
    normalizationContext: BroadcastWrapper[NormalizationContext],
    maxNumImprovementFailures: Int = TRON.DEFAULT_MAX_NUM_FAILURE,
    tolerance: Double = TRON.DEFAULT_TOLERANCE,
    maxNumIterations: Int = TRON.DEFAULT_MAX_ITER,
    constraintMap: Option[Map[Int, (Double, Double)]] = Optimizer.DEFAULT_CONSTRAINT_MAP)
  extends Optimizer[TwiceDiffFunction](
    tolerance,
    maxNumIterations,
    normalizationContext,
    constraintMap) {

  /**
   * Initialize the hyperparameters for TRON (see Reference 2 for more details).
   */
  private val (eta0, eta1, eta2) = (1e-4, 0.25, 0.75)
  private val (sigma1, sigma2, sigma3) = (0.25, 0.5, 4.0)

  /**
   * delta is the trust region size.
   */
  private var delta = Double.MaxValue

  /**
   * Initialize the trust region size.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param initState The initial state of the optimizer
   * @param data The training data
   */
  def init(objectiveFunction: TwiceDiffFunction, initState: OptimizerState)(data: objectiveFunction.Data): Unit =
    delta = norm(initState.gradient, 2)

  /**
   * Get the optimizer's state.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param coefficients The model coefficients
   * @param iter The current iteration of the optimizer
   * @param data The training data
   * @return The current optimizer state
   */
  protected def calculateState(
      objectiveFunction: TwiceDiffFunction,
      coefficients: Vector[Double],
      iter: Int = 0)(
      data: objectiveFunction.Data): OptimizerState = {

    val (value, gradient) = objectiveFunction.calculate(data, coefficients, normalizationContext)

    OptimizerState(coefficients, value, gradient, iter)
  }

  /**
   * Reset the delta.
   */
  override def clearOptimizerInnerState(): Unit = {

    super.clearOptimizerInnerState()
    delta = Double.MaxValue
  }

  /**
   * Run one iteration of the optimizer given the current state.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param currState The current optimizer state
   * @param data The training data
   * @return The updated state of the optimizer
   */
  protected def runOneIteration(
      objectiveFunction: TwiceDiffFunction,
      currState: OptimizerState)(
      data: objectiveFunction.Data): OptimizerState = {

    val prevCoefficients = currState.coefficients
    val prevFunctionValue = currState.loss
    val prevFunctionGradient = currState.gradient
    val prevIter = currState.iter
    val isFirstIteration = currState.iter == 0

    var improved = false
    var numImprovementFailure = 0
    var finalState = currState
    do {
      // Retry the TRON optimization with the shrunken trust region boundary (delta) until either:
      // 1. The function value is improved
      // 2. The maximum number of improvement failures reached.
      val (cgIter, step, residual) = TRON.truncatedConjugateGradientMethod(
        objectiveFunction,
        prevCoefficients,
        prevFunctionGradient,
        normalizationContext,
        delta)(
        data)

      val updatedCoefficients = prevCoefficients + step
      val gs = prevFunctionGradient.dot(step)
      // Compute the predicted reduction
      val predictedReduction = -0.5 * (gs - step.dot(residual))
      // Function value
      val (updatedFunctionValue, updatedFunctionGradient) = objectiveFunction.calculate(
        data,
        updatedCoefficients,
        normalizationContext)

      // Compute the actual reduction.
      val actualReduction = prevFunctionValue - updatedFunctionValue
      val stepNorm = norm(step, 2)

      // On the first iteration, adjust the initial step bound.
      if (isFirstIteration) {
        delta = math.min(delta, stepNorm)
      }

      // Compute prediction alpha*stepNorm of the step.
      val alpha = if (updatedFunctionValue - prevFunctionValue - gs <= 0) {
          sigma3
        } else {
          math.max(sigma1, -0.5 * (gs / (updatedFunctionValue - prevFunctionValue - gs)))
        }

      // Update the trust region bound according to the ratio of actual to predicted reduction.
      if (actualReduction < eta0 * predictedReduction) {
        delta = math.min(math.max(alpha, sigma1) * stepNorm, sigma2 * delta)
      } else if (actualReduction < eta1 * predictedReduction) {
        delta = math.max(sigma1 * delta, math.min(alpha * stepNorm, sigma2 * delta))
      } else if (actualReduction < eta2 * predictedReduction) {
        delta = math.max(sigma1 * delta, math.min(alpha * stepNorm, sigma3 * delta))
      } else {
        delta = math.max(delta, math.min(alpha * stepNorm, sigma3 * delta))
      }
      val gradientNorm = norm(updatedFunctionGradient, 2)
      val residualNorm = norm(residual, 2)
      logger.debug(f"iter $prevIter%3d act $actualReduction%5.3e pre $predictedReduction%5.3e delta $delta%5.3e " +
        f"f $updatedFunctionValue%5.3e |residual| $residualNorm%5.3e |g| $gradientNorm%5.3e CG $cgIter%3d")

      if (actualReduction > eta0 * predictedReduction) {
        // if the actual function value reduction is greater than eta0 times the predicted function value reduction,
        // we accept the updated coefficients and move forward with the updated optimizer state
        val coefficients = updatedCoefficients

        improved = true
        /* project coefficients into constrained space, if any, after the optimization step */
        finalState = OptimizerState(
          OptimizationUtils.projectCoefficientsToSubspace(coefficients, constraintMap),
          updatedFunctionValue,
          updatedFunctionGradient,
          prevIter + 1)
      } else {
        // otherwise, the updated coefficients will not be accepted, and the old state will be returned along with
        // warning messages
        logger.warn(s"actual objective function value reduction is smaller than predicted " +
          s"(actualReduction = $actualReduction < eta0 = $eta0 * predictedReduction = $predictedReduction)")
        if (updatedFunctionValue < -1.0e+32) {
          logger.warn("updated function value < -1.0e+32")
        }
        if (actualReduction <= 0) {
          logger.warn("actual reduction of function value <= 0")
        }
        if (math.abs(actualReduction) <= 1.0e-12 && math.abs(predictedReduction) <= 1.0e-12) {
          logger.warn("both actual reduction and predicted reduction of function value are too small")
        }
        numImprovementFailure += 1
      }
    } while (!improved && numImprovementFailure < maxNumImprovementFailures)

    finalState
  }
}

object TRON extends Logging {

  val DEFAULT_MAX_NUM_FAILURE = 5
  val DEFAULT_TOLERANCE = 1.0E-5
  val DEFAULT_MAX_ITER = 15
  // The maximum number of iterations used in the conjugate gradient update. Larger value will lead to more accurate
  // solution but also longer running time.
  val MAX_CG_ITERATIONS: Int = 20

  /**
   * Run the truncated conjugate gradient (CG) method as a subroutine of TRON.
   * For details and notations of the following code, please see Algorithm 2
   * (Conjugate gradient procedure for approximately solving the trust region sub-problem)
   * in page 6 of the following paper: [[http://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf]].
   *
   * @param objectiveFunction The objective function
   * @param coefficients The model coefficients
   * @param gradient Gradient of the objective function
   * @param truncationBoundary The truncation boundary of truncatedConjugateGradientMethod.
   *                           In the case of Tron, this corresponds to the trust region size (delta).
   * @param data The training data
   * @return Tuple3(number of CG iterations, solution, residual)
   */
  private def truncatedConjugateGradientMethod(
      objectiveFunction: TwiceDiffFunction,
      coefficients: Vector[Double],
      gradient: Vector[Double],
      normalizationContext: BroadcastWrapper[NormalizationContext],
      truncationBoundary: Double)
      (data: objectiveFunction.Data): (Int, Vector[Double], Vector[Double]) = {

    val step = VectorUtils.zeroOfSameType(gradient)
    val residual = gradient * -1.0
    val direction = residual.copy
    val conjugateGradientConvergenceTolerance = 0.1 * norm(gradient, 2)
    var iteration = 0
    var done = false
    var rTr = residual.dot(residual)
    while (iteration < MAX_CG_ITERATIONS && !done) {
      if (norm(residual, 2) <= conjugateGradientConvergenceTolerance) {
        done = true
      } else {
        iteration += 1
        // Compute the hessianVector
        val Hd = objectiveFunction.hessianVector(data, coefficients, direction, normalizationContext)
        var alpha = rTr / direction.dot(Hd)

        step += direction * alpha
        if (norm(step, 2) > truncationBoundary) {
          logger.debug(s"cg reaches truncation boundary after $iteration iterations")
          // Solve equation (13) of Algorithm 2
          alpha = -alpha
          step += direction * alpha
          val std = step.dot(direction)
          val sts = step.dot(step)
          val dtd = direction.dot(direction)
          val dsq = truncationBoundary * truncationBoundary
          val rad = math.sqrt(std * std + dtd * (dsq - sts))
          if (std >= 0) {
            alpha = (dsq - sts) / (std + rad)
          } else {
            alpha = (rad - std) / dtd
          }
          step += direction * alpha
          alpha = -alpha
          residual += Hd * alpha
          done = true
        } else {
          // Find the new conjugate gradient direction
          alpha = -alpha
          residual += Hd * alpha
          val rnewTrnew = residual.dot(residual)
          val beta = rnewTrnew / rTr
          direction := direction * beta + residual
          rTr = rnewTrnew
        }
      }
    }

    (iteration, step, residual)
  }
}
