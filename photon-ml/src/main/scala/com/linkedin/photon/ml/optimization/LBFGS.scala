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
package com.linkedin.photon.ml.optimization

import breeze.linalg.Vector
import breeze.optimize.{DiffFunction => BreezeDiffFunction, LBFGS => BreezeLBFGS}
import org.apache.spark.broadcast.Broadcast

import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization.NormalizationContext

/**
 * Class used to solve an optimization problem using Limited-memory BFGS (LBFGS).
 * Reference: [[http://en.wikipedia.org/wiki/Limited-memory_BFGS]].
 *
 * @param normalizationContext The normalization context
 * @param tolerance The tolerance threshold for improvement between iterations as a percentage of the initial loss
 * @param maxNumIterations The cut-off for number of optimization iterations to perform.
 * @param numCorrections The number of corrections used in the LBFGS update. Default 10. Values of numCorrections less
 *                       than 3 are not recommended; large values of numCorrections will result in excessive computing
 *                       time.
 *                       Recommended:  3 < numCorrections < 10
 *                       Restriction:  numCorrections > 0
 * @param constraintMap (Optional) The map of constraints on the feature coefficients
 * @param isTrackingState Whether to track intermediate states during optimization
 * @param isReusingPreviousInitialState Whether to reuse the previous initial state or not. When warm-start training is
 *                                      desired, i.e. in grid-search based hyper-parameter tuning, this field is
 *                                      recommended to set to true for consistent convergence check.
 */
class LBFGS(
    normalizationContext: Broadcast[NormalizationContext],
    numCorrections: Int = LBFGS.DEFAULT_NUM_CORRECTIONS,
    tolerance: Double = LBFGS.DEFAULT_TOLERANCE,
    maxNumIterations: Int = LBFGS.DEFAULT_MAX_ITER,
    constraintMap: Option[Map[Int, (Double, Double)]] = Optimizer.DEFAULT_CONSTRAINT_MAP,
    isTrackingState: Boolean = Optimizer.DEFAULT_TRACKING_STATE,
    isReusingPreviousInitialState: Boolean = Optimizer.DEFAULT_REUSE_PREVIOUS_INIT_STATE)
  extends Optimizer[DiffFunction](
    tolerance,
    maxNumIterations,
    normalizationContext,
    constraintMap,
    isTrackingState,
    isReusingPreviousInitialState) {

  /**
   * Under the hood, this adaptor uses an LBFGS
   * ([[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.LBFGS breeze.optimize.LBFGS]]) optimizer from
   * Breeze to optimize functions. The DiffFunction is modified into a Breeze DiffFunction which the Breeze optimizer
   * can understand.
   */
  protected class BreezeOptimization(
      diffFunction: BreezeDiffFunction[Vector[Double]],
      initCoefficients: Vector[Double]) {

    // The actual workhorse
    private val breezeStates = breezeOptimizer.iterations(diffFunction, initCoefficients)
    breezeStates.next()

    def next(state: OptimizerState): OptimizerState = {
      if (breezeStates.hasNext) {
        val breezeState = breezeStates.next()
        // Project coefficients into constrained space, if any, before updating the state
        OptimizerState(
          OptimizationUtils.projectCoefficientsToHypercube(breezeState.x, constraintMap),
          breezeState.adjustedValue,
          breezeState.adjustedGradient,
          state.iter + 1)

      } else {
        // LBFGS is converged
        state
      }
    }
  }

  protected val breezeOptimizer = new BreezeLBFGS[Vector[Double]](maxNumIterations, numCorrections, tolerance)
  @transient
  protected var breezeOptimization: BreezeOptimization = _

  /**
   * Initialize breeze optimization engine.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param initState The initial state of the optimizer, prior to starting optimization
   * @param data The training data
   */
  override def init(objectiveFunction: DiffFunction, initState: OptimizerState)(data: objectiveFunction.Data): Unit = {
    val breezeDiffFunction = new BreezeDiffFunction[Vector[Double]]() {
        // Calculating the gradient and value of the objective function
        override def calculate(coefficients: Vector[Double]): (Double, Vector[Double]) = {
          val convertedCoefficients = objectiveFunction.convertFromVector(coefficients)
          val result = objectiveFunction.calculate(data, convertedCoefficients, normalizationContext)

          objectiveFunction.cleanupCoefficients(convertedCoefficients)
          result
        }
      }
    breezeOptimization = new BreezeOptimization(breezeDiffFunction, initState.coefficients)
  }

  /**
   * Get the optimizer's state
   *
   * @param objectiveFunction The objective function to be optimized
   * @param coefficients The model coefficients
   * @param iter The current iteration of the optimizer
   * @param data The training data
   * @return The current optimizer state
   */
  override protected def getState
    (objectiveFunction: DiffFunction, coefficients: Vector[Double], iter: Int = 0)
    (data: objectiveFunction.Data) : OptimizerState = {

    val convertedCoefficients = objectiveFunction.convertFromVector(coefficients)
    val (value, gradient) = objectiveFunction.calculate(data, convertedCoefficients, normalizationContext)

    objectiveFunction.cleanupCoefficients(convertedCoefficients)
    OptimizerState(coefficients, value, gradient, iter)
  }

  /**
   * Clear the [[OptimizationStatesTracker]]
   */
  override def clearOptimizerInnerState(): Unit = breezeOptimization = _: BreezeOptimization

  /**
   * Run one iteration of the optimizer given the current state
   *
   * @param objectiveFunction The objective function to be optimized
   * @param currState The current optimizer state
   * @param data The training data
   * @return The updated state of the optimizer
   */
  override protected def runOneIteration
    (objectiveFunction: DiffFunction, currState: OptimizerState)
    (data: objectiveFunction.Data): OptimizerState = breezeOptimization.next(currState)
}

object LBFGS {
  val DEFAULT_MAX_ITER = 100
  val DEFAULT_NUM_CORRECTIONS = 10
  val DEFAULT_TOLERANCE = 1.0E-7
}
