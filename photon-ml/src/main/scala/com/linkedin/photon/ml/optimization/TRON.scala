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

/*
 * NOTE: The codes are heavily influenced by SPARK LIBLINEAR TRON implementation,
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
import com.linkedin.photon.ml.data.DataPoint
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.util.Utils
import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

/**
 * This class used to solve an optimization problem using trust region Newton method (TRON).
 * Reference1: [[http://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf]]
 * Reference2:
 *   [[http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/distributed-liblinear/spark/running_spark_liblinear.html]]
 * @tparam Datum Generic type of input data point
 *
 * @param maxNumImprovementFailures
 * The maximum allowed number of times in a row the objective hasn't improved.
 * For most optimizers using line search, like L-BFGS, the improvement failure is not supposed to happen,
 * because any improvement failure should be captured during the line search step.
 * And here we are trying to capture the improvement failure after the gradient step. As a result,
 * any improvement failure in this cases should be resulted by some bug and we should not tolerate it.
 * However, for optimizers like TRON, maxNumImprovementFailures is set to 5, because occasional improvement failure is
 * acceptable.
 */
class TRON[Datum <: DataPoint](
    var maxNumImprovementFailures: Int = TRON.DEFAULT_MAX_NUM_FAILURE)
  extends AbstractOptimizer[Datum, TwiceDiffFunction[Datum]](
    tolerance = TRON.DEFAULT_TOLERANCE, maxNumIterations = TRON.DEFAULT_MAX_ITER) {

  /**
   * Customized optimization parameter for Tron
   */
  maxNumIterations = TRON.DEFAULT_MAX_ITER
  tolerance = TRON.DEFAULT_TOLERANCE


  /**
   * Initialize the hyper-parameters for Tron. See the Reference2 for more details on the hyper-parameters.
   */
  private val (eta0, eta1, eta2) = (1e-4, 0.25, 0.75)
  private val (sigma1, sigma2, sigma3) = (0.25, 0.5, 4.0)

  /**
   * delta is the trust region size.
   */
  private var delta = Double.MaxValue

  def init(
      state: OptimizerState,
      data: Either[RDD[Datum], Iterable[Datum]],
      diffFunction: TwiceDiffFunction[Datum],
      initialCoef: Vector[Double]) {
    delta = norm(state.gradient, 2)
  }

  override def clearOptimizerInnerState() {
    delta = Double.MaxValue
  }

  def runOneIteration(
      dataPoints: Either[RDD[Datum], Iterable[Datum]],
      objectiveFunction: TwiceDiffFunction[Datum],
      state: OptimizerState): OptimizerState = {

    val distributedOptimization = dataPoints.isLeft
    val lastCoefficients =
      if (distributedOptimization) {
        Left(dataPoints.left.get.sparkContext.broadcast(state.coefficients))
      } else {
        Right(state.coefficients)
      }
    val lastFunctionValue = state.value
    val lastFunctionGradient = state.gradient
    val isFirstIteration = state.iter == 0

    var improved = false
    var numImprovementFailure = 0
    var finalState = state
    do {
      // retry the TRON optimization with the shrunken trust region boundary (delta) until either
      // 1. the function value is improved; or
      // 2. the maximum number of improvement failures reached.
      val (cgIter, step, residual) = TRON.truncatedConjugateGradientMethod(
        dataPoints, objectiveFunction, lastCoefficients, lastFunctionGradient, delta)

      val updatedCoefficients =
        if (distributedOptimization) {
          val updated = dataPoints.left.get.sparkContext.broadcast(lastCoefficients.left.get.value + step)
          lastCoefficients.left.get.unpersist()
          Left(updated)
        } else {
          val updated = lastCoefficients.right.get + step
          Right(updated)
        }
      val gs = lastFunctionGradient.dot(step)
      /* Compute the predicted reduction */
      val predictedReduction = -0.5 * (gs - step.dot(residual))
      /* Function value */
      val (updatedFunctionValue, updatedFunctionGradient) =
        if (distributedOptimization) {
          objectiveFunction.calculate(dataPoints.left.get, updatedCoefficients.left.get)
        } else {
          objectiveFunction.calculate(dataPoints.right.get, updatedCoefficients.right.get)
        }

      /* Compute the actual reduction. */
      val actualReduction = lastFunctionValue - updatedFunctionValue
      val stepNorm = norm(step, 2)

      /* On the first iteration, adjust the initial step bound. */
      if (isFirstIteration) delta = math.min(delta, stepNorm)

      /* Compute prediction alpha*stepNorm of the step. */
      val alpha =
        if (updatedFunctionValue - lastFunctionValue - gs <= 0) {
          sigma3
        } else {
          math.max(sigma1, -0.5 * (gs / (updatedFunctionValue - lastFunctionValue - gs)))
        }

      /* Update the trust region bound according to the ratio of actual to predicted reduction. */
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
      logDebug(f"iter ${state.iter}%3d act $actualReduction%5.3e pre $predictedReduction%5.3e delta $delta%5.3e " +
        f"f $updatedFunctionValue%5.3e |residual| $residualNorm%5.3e |g| $gradientNorm%5.3e CG $cgIter%3d")

      if (actualReduction > eta0 * predictedReduction) {
        // if the actual function value reduction is greater than eta0 times the predicted function value reduction,
        // we accept the updated coefficients and move forward with the updated optimizer state
        val coefficients =
          if (distributedOptimization) {
            updatedCoefficients.left.get.value
          } else {
            updatedCoefficients.right.get
          }

        improved = true
        /* project coefficients into constrained space, if any, after the optimization step */
        val projectedCoefficients = OptimizationUtils.projectCoefficientsToHypercube(coefficients, constraintMap)
        finalState = OptimizerState(projectedCoefficients, updatedFunctionValue, updatedFunctionGradient,
          state.iter + 1)
      } else {
        // otherwise, the updated coefficients will not be accepted, and the old state will be returned along with
        // warning messages
        logWarning(s"actual objective function value reduction is smaller than predicted " +
          s"(actualReduction = $actualReduction < eta0 = $eta0 * predictedReduction = $predictedReduction)")
        if (updatedFunctionValue < -1.0e+32) {
          logWarning("updated function value < -1.0e+32")
        }
        if (actualReduction <= 0) {
          logWarning("actual reduction of function value <= 0")
        }
        if (math.abs(actualReduction) <= 1.0e-12 && math.abs(predictedReduction) <= 1.0e-12) {
          logWarning("both actual reduction and predicted reduction of function value are too small")
        }
        numImprovementFailure += 1
      }
    } while (!improved && numImprovementFailure < maxNumImprovementFailures)
    finalState
  }
}

object TRON extends Logging {
  val DEFAULT_MAX_ITER = 15
  val DEFAULT_TOLERANCE = 1.0E-5
  val DEFAULT_MAX_NUM_FAILURE = 5
  /**
   * The maximum number of iterations used in the conjugate gradient update. Larger value will lead to
   * more accurate solution but also longer running time. Default: 20.
   */
  var maxNumCGIterations: Int = 20

  /**
   * Run the truncated conjugate gradient (CG) method as a subroutine of TRON.
   * For details and notations of the following code, please see Algorithm 2
   * (Conjugate gradient procedure for approximately solving the trust region sub-problem)
   * in page 6 of the following paper: [[http://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf]]
   * @param dataPoints Training data points
   * @param objectiveFunction The objective function
   * @param coefficients The model coefficients
   * @param gradient Gradient of the objective function
   * @param truncationBoundary The truncation boundary of truncatedConjugateGradientMethod.
   *                           In the case of Tron, this corresponds to the trust region size (delta).
   * @return Tuple3(number of CG iterations, solution, residual)
   */
  private def truncatedConjugateGradientMethod[Datum <: DataPoint](
      dataPoints: Either[RDD[Datum], Iterable[Datum]],
      objectiveFunction: TwiceDiffFunction[Datum],
      coefficients: Either[Broadcast[Vector[Double]], Vector[Double]],
      gradient: Vector[Double],
      truncationBoundary: Double): (Int, Vector[Double], Vector[Double]) = {

    val step = Utils.initializeZerosVectorOfSameType(gradient)
    val residual = gradient * -1.0
    val direction = residual.copy
    val conjugateGradientConvergenceTolerance = 0.1 * norm(gradient, 2)
    var iteration = 0
    var done = false
    var rTr = residual.dot(residual)
    while (iteration < maxNumCGIterations && !done) {
      if (norm(residual, 2) <= conjugateGradientConvergenceTolerance) {
        done = true
      } else {
        iteration += 1
        /* compute the hessianVector */
        val Hd = {
          dataPoints match {
            //The calculation is done in a distributed fashion
            case Left(dataPointsRDD) =>
              val broadcastedDirection = dataPointsRDD.sparkContext.broadcast(direction)
              val hessianVector = objectiveFunction.hessianVector(
                dataPointsRDD, coefficients.left.get, broadcastedDirection)
              broadcastedDirection.unpersist()
              hessianVector
            //The calculation is done on a local machine
            case Right(dataPointsIterable) =>
              objectiveFunction.hessianVector(dataPointsIterable, coefficients.right.get, direction)
          }
        }
        var alpha = rTr / direction.dot(Hd)
        step += direction * alpha
        if (norm(step, 2) > truncationBoundary) {
          logDebug(s"cg reaches truncation boundary after $iteration iterations")
          /* Solve equation (13) of Algorithm 2 */
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
          /* find the new conjugate gradient direction */
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
