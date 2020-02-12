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
package com.linkedin.photon.ml.optimization

import breeze.linalg.{norm, Vector}
import breeze.numerics.abs

import com.linkedin.photon.ml.function.ObjectiveFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util._

/**
 * Common base class for the Photon ML optimization problem solvers.
 *
 * @tparam Function Generic type of the objective function to be optimized
 * @param relTolerance The relative tolerance used to gauge improvement between iterations.
 *                     Separate absolute tolerances for the loss function and the gradient are set from this relative
 *                     tolerance.
 * @param maxNumIterations The max number of iterations to perform
 * @param normalizationContext The normalization context
 * @param constraintMap (Optional) The map of constraints on the feature coefficients
 */
abstract class Optimizer[-Function <: ObjectiveFunction](
    relTolerance: Double,
    maxNumIterations: Int,
    normalizationContext: BroadcastWrapper[NormalizationContext],
    constraintMap: Option[Map[Int, (Double, Double)]])
  extends Serializable
  with Logging {

  /**
   * Tracks all the intermediate states of the optimizer and times the optimization.
   */
  private var statesTracker: OptimizationStatesTracker = _

  /**
   * The absolute tolerances for the optimization problem.
   */
  private var lossAbsTolerance: Double = 0D
  private var gradientAbsTolerance: Double = 0D

  /**
   * The current and previous states of the Optimizer: coefficients, loss, gradient and iteration number.
   */
  private var currentState: Option[OptimizerState] = None
  private var previousState: Option[OptimizerState] = None

  /**
   * Set the absolute tolerances for the optimizer from the loss and gradient values in the provided state.
   *
   * @param state The initial state
   */
  private def setAbsTolerances(state: OptimizerState): Unit = {

    lossAbsTolerance = state.loss * relTolerance
    gradientAbsTolerance = norm(state.gradient, 2) * relTolerance
  }

  /**
   * Update the current and previous states for the optimizer.
   *
   * @param state The current state
   */
  private def updateCurrentState(state: OptimizerState): Unit = {

    currentState match {
      case Some(currState) =>
        // Only tracks the state if it is different from the previous state (e.g., different objects)
        if (state != currState) {
          statesTracker.track(state)
        }

      case None => statesTracker.track(state)
    }

    previousState = currentState
    currentState = Some(state)
  }

  /**
   * Get the normalization context.
   */
  def getNormalizationContext: BroadcastWrapper[NormalizationContext] = normalizationContext

  /**
   * Get the state tracker.
   */
  def getStateTracker: OptimizationStatesTracker = statesTracker

  /**
   * Get the current state of the optimizer.
   *
   * @return The current state of the optimizer
   */
  def getCurrentState: Option[OptimizerState] = currentState

  /**
   * Get the previous state of the optimizer.
   *
   * @return The previous state of the optimizer
   */
  def getPreviousState: Option[OptimizerState] = previousState

  /**
   * Get the optimizer convergence reason.
   *
   * @note It is not strictly necessary to check both the convergence of the loss function and the convergence of the
   *       gradient, from a correctness point of view. All we need in the end is convergence of the loss function to
   *       its optimum value. However, it can be useful to have a stopping criterion based on the gradient norm as
   *       the gradient can "misbehave" around the optimum of the loss function (oscillations, numerical issues...).
   *
   * @return The convergence reason
   */
  def getConvergenceReason: Option[ConvergenceReason] =
    if (currentState.isEmpty) {
      None
    } else if (currentState.get.iter >= maxNumIterations) {
      Some(MaxIterations)
    } else if (currentState.get.iter == previousState.get.iter) {
      Some(ObjectiveNotImproving)
    } else if (abs(currentState.get.loss - previousState.get.loss) <= lossAbsTolerance) {
      Some(FunctionValuesConverged)
    } else if (norm(currentState.get.gradient, 2) <= gradientAbsTolerance) {
      Some(GradientConverged)
    } else {
      None
    }

  /**
   * Check whether optimization has completed.
   *
   * @return True if the optimizer thinks it's done, false otherwise
   */
  def isDone: Boolean = getConvergenceReason.nonEmpty

  /**
   * Solve the provided convex optimization problem.
   *
   * @note This function is called for each regularization weight separately.
   * @note For cold start, initialCoefficients are always zero, so that the first current state for each regularization
   *       weight is always computed from 0.
   *       For warm start, initialCoefficients contains the coefficients of the model optimized for the previous
   *       regularization weight.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param initialCoefficients The initial coefficients from which to begin optimization
   * @param data The training data
   * @return Optimized model coefficients and corresponding objective function's value
   */
  protected[ml] def optimize(
      objectiveFunction: Function,
      initialCoefficients: Vector[Double])(
      data: objectiveFunction.Data): (Vector[Double], OptimizationStatesTracker) = {

    val normalizedInitialCoefficients = normalizationContext.value.modelToTransformedSpace(initialCoefficients)

    // Reset Optimizer inner state
    clearOptimizerInnerState()

    // We set the absolute tolerances from the magnitudes of the first loss and gradient
    setAbsTolerances(calculateState(objectiveFunction, VectorUtils.zeroOfSameType(normalizedInitialCoefficients))(data))

    // TODO: For cold start, we call calculateState with the same arguments twice
    val initialState = calculateState(objectiveFunction, normalizedInitialCoefficients)(data)
    init(objectiveFunction, initialState)(data)
    updateCurrentState(initialState)

    do {
      updateCurrentState(runOneIteration(objectiveFunction, getCurrentState.get)(data))
    } while (!isDone)

    statesTracker.convergenceReason = getConvergenceReason
    val currState = getCurrentState.get
    (currState.coefficients, statesTracker)
  }

  /**
   * Initialize the context of the optimizer (e.g., the history of LBFGS; the trust region size of TRON; etc.).
   *
   * @param objectiveFunction The objective function to be optimized
   * @param initState The initial state of the optimizer
   * @param data The training data
   */
  protected def init(objectiveFunction: Function, initState: OptimizerState)(data: objectiveFunction.Data): Unit

  /**
   * Calculate the Optimizer state given some data.
   *
   * @note involves a calculation over the whole dataset, so can be expensive.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param coefficients The model coefficients
   * @param iter The current iteration of the optimizer
   * @param data The training data
   * @return The current optimizer state
   */
  protected def calculateState(
      objectiveFunction: Function,
      coefficients: Vector[Double],
      iter: Int = 0)(
      data: objectiveFunction.Data): OptimizerState

  /**
   * Clear the optimizer (e.g. the history of LBFGS; the trust region size of TRON; etc.).
   *
   * @note This function should be protected and not exposed.
   */
  protected def clearOptimizerInnerState(): Unit = {

    currentState = None
    previousState = None
    statesTracker = new OptimizationStatesTracker()
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
      objectiveFunction: Function,
      currState: OptimizerState)(
      data: objectiveFunction.Data): OptimizerState
}

object Optimizer {

  val DEFAULT_CONSTRAINT_MAP: Option[Map[Int, (Double, Double)]] = None
}
