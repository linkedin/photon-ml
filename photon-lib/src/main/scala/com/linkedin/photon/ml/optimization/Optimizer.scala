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

import breeze.linalg.{Vector, norm}
import breeze.numerics.abs
import org.apache.spark.broadcast.Broadcast

import com.linkedin.photon.ml.function.ObjectiveFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.{
  ConvergenceReason, FunctionValuesConverged, GradientConverged, MaxIterations, ObjectiveNotImproving, Logging}

/**
 * Common base class for the Photon ML optimization problem solvers.
 *
 * @param tolerance The tolerance threshold for improvement between iterations as a percentage of the initial loss
 * @param maxNumIterations The cut-off for number of optimization iterations to perform.
 * @param normalizationContext The normalization context
 * @param constraintMap (Optional) The map of constraints on the feature coefficients
 * @param isTrackingState Whether to track intermediate states during optimization
 * @param isReusingPreviousInitialState Whether to reuse the previous initial state or not. When warm-start training is
 *                                      desired, i.e. in grid-search based hyper-parameter tuning, this field is
 *                                      recommended to set to true for consistent convergence check.
 * @tparam Function Generic type of the objective function to be optimized.
 */
abstract class Optimizer[-Function <: ObjectiveFunction](
    tolerance: Double,
    maxNumIterations: Int,
    normalizationContext: Broadcast[NormalizationContext],
    constraintMap: Option[Map[Int, (Double, Double)]],
    isTrackingState: Boolean,
    isReusingPreviousInitialState: Boolean)
  extends Serializable
  with Logging {

  protected var statesTracker: Option[OptimizationStatesTracker] = if (isTrackingState) {
    Some(new OptimizationStatesTracker())
  } else {
    None
  }
  protected var initialState: Option[OptimizerState] = None
  protected var currentState: Option[OptimizerState] = None
  protected var previousState: Option[OptimizerState] = None

  /**
   * Initialize the context of the optimizer (e.g., the history of LBFGS; the trust region size of TRON; etc.).
   *
   * @param objectiveFunction The objective function to be optimized
   * @param initState The initial state of the optimizer
   * @param data The training data
   */
  protected def init(objectiveFunction: Function, initState: OptimizerState)(data: objectiveFunction.Data): Unit

  /**
   * Get the normalization context.
   */
  def getNormalizationContext: Broadcast[NormalizationContext] = normalizationContext

  /**
   * Get the state tracker.
   */
  def getStateTracker: Option[OptimizationStatesTracker] = statesTracker

  /**
   * Get the optimizer's state.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param coefficients The model coefficients
   * @param iter The current iteration of the optimizer
   * @param data The training data
   * @return The current optimizer state
   */
  protected def getState
    (objectiveFunction: Function, coefficients: Vector[Double], iter: Int = 0)
    (data: objectiveFunction.Data): OptimizerState

  /**
   * Get the initial state of the optimizer (used for checking convergence).
   *
   * @return The initial state of the optimizer
   */
  def getInitialState: Option[OptimizerState] = initialState

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
   * Set the initial state for the optimizer.
   *
   * @param state The initial state
   */
  protected def setInitialState(state: OptimizerState): Unit = {
    initialState = Some(state)
  }

  /**
   * Update the current and previous states for the optimizer.
   *
   * @param state The current state
   */
  protected def updateCurrentState(state: OptimizerState): Unit = {
    (statesTracker, currentState) match {
      case (Some(tracker), Some(currState)) =>
        // Only tracks the state if it is different from the previous state (e.g., different objects)
        if (state != currState) {
          tracker.track(state)
        }
      case _ =>
    }
    previousState = currentState
    currentState = Some(state)
  }

  /**
   * Clear the [[OptimizationStatesTracker]].
   */
  protected def clearOptimizationStatesTracker(): Unit =
    statesTracker = statesTracker.map(tracker => new OptimizationStatesTracker())

  /**
   * Clear the optimizer (e.g. the history of LBFGS; the trust region size of TRON; etc.).
   *
   * @note This function should be protected and not exposed.
   */
  protected def clearOptimizerInnerState(): Unit

  /**
   * Get the optimizer convergence reason.
   *
   * @return The convergence reason
   */
  def getConvergenceReason: Option[ConvergenceReason] = {
    if (initialState.isEmpty || currentState.isEmpty || previousState.isEmpty) {
      None
    } else if (currentState.get.iter >= maxNumIterations) {
      Some(MaxIterations)
    } else if (currentState.get.iter == previousState.get.iter) {
      Some(ObjectiveNotImproving)
    } else if (abs(currentState.get.value - previousState.get.value) <= tolerance * initialState.get.value) {
      Some(FunctionValuesConverged)
    } else if (norm(currentState.get.gradient, 2) <= tolerance * norm(initialState.get.gradient, 2)) {
      Some(GradientConverged)
    } else {
      None
    }
  }

  /**
   * Determine the convergence reason once optimization is complete.
   */
  protected def determineConvergenceReason(): Unit = {
    // Set the convergenceReason when the optimization is done
    statesTracker match {
      case Some(y) => y.convergenceReason = getConvergenceReason
      case None =>
    }
  }

  /**
   * Check whether optimization has completed.
   *
   * @return True if the optimizer thinks it's done, false otherwise
   */
  def isDone: Boolean = getConvergenceReason.nonEmpty

  /**
   * Run one iteration of the optimizer given the current state.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param currState The current optimizer state
   * @param data The training data
   * @return The updated state of the optimizer
   */
  protected def runOneIteration
    (objectiveFunction: Function, currState: OptimizerState)
    (data: objectiveFunction.Data): OptimizerState

  /**
   * Solve the provided convex optimization problem.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param data The training data
   * @return Optimized coefficients and the optimized objective function's value
   */
  protected[ml] def optimize(objectiveFunction: Function)(data: objectiveFunction.Data): (Vector[Double], Double) = {
    val initialCoefficients = Vector.zeros[Double](objectiveFunction.domainDimension(data))
    optimize(objectiveFunction, initialCoefficients)(data)
  }

  /**
   * Solve the provided convex optimization problem.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param initialCoefficients Initial coefficients
   * @param data The training data
   * @return Optimized coefficients and the optimized objective function's value
   */
  protected[ml] def optimize
    (objectiveFunction: Function, initialCoefficients: Vector[Double])
    (data: objectiveFunction.Data): (Vector[Double], Double) = {

    clearOptimizerInnerState()
    clearOptimizationStatesTracker()

    val startState = getState(objectiveFunction, initialCoefficients)(data)
    updateCurrentState(startState)
    // Initialize the optimizer state if it's not being initialized yet, or if we don't need to reuse the existing
    // initial state for consistent convergence check across multiple runs.
    if (getInitialState.isEmpty || !isReusingPreviousInitialState) {
      setInitialState(startState)
    }
    init(objectiveFunction, startState)(data)

    do {
      updateCurrentState(runOneIteration(objectiveFunction, getCurrentState.get)(data))
    } while (!isDone)

    determineConvergenceReason()
    val currState = getCurrentState
    (currState.get.coefficients, currState.get.value)
  }
}

object Optimizer {
  val DEFAULT_CONSTRAINT_MAP = None
  val DEFAULT_TRACKING_STATE = true
  val DEFAULT_REUSE_PREVIOUS_INIT_STATE = true
}
