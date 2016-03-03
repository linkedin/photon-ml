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

import breeze.linalg.norm
import breeze.numerics._
import breeze.optimize.FirstOrderMinimizer._
import com.linkedin.photon.ml.data.DataPoint
import com.linkedin.photon.ml.function.DiffFunction

/**
 * Common book-keeping for the states of implementing [[Optimizer]] and its convergence check
 *
 * @author bdrew
 */
abstract class AbstractOptimizer[Datum <: DataPoint, -Function <: DiffFunction[Datum]](
    protected var tolerance: Double = 1e-6,
    protected var maxNumIterations: Int = 80,
    protected var constraintMap: Option[Map[Int, (Double, Double)]] = None,
    protected var isTrackingState: Boolean = true)
  extends Optimizer[Datum, Function] {

  protected var initialState: Option[OptimizerState] = None

  protected var currentState: Option[OptimizerState] = None

  protected var previousState: Option[OptimizerState] = None

  protected var statesTracker: Option[OptimizationStatesTracker] = if (isTrackingState) {
    Some(new OptimizationStatesTracker())
  } else {
    None
  }

  def isDone: Boolean = convergenceReason.nonEmpty

  def convergenceReason: Option[ConvergenceReason] = {
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

  def getTolerance: Double = tolerance

  def setTolerance(tol: Double) {
    require(!(tol.isInfinite || tol.isNaN) && tol > 0.0)
    tolerance = tol
  }

  def getMaximumIterations: Int = maxNumIterations

  def setMaximumIterations(maxIter: Int) {
    require(maxIter > 0)
    maxNumIterations = maxIter
  }

  def getConstraintMap: Option[Map[Int, (Double, Double)]] = constraintMap

  def setConstraintMap(constraints: Option[Map[Int, (Double, Double)]]) {
    constraintMap = constraints
  }

  def setStateTrackingEnabled(enabled: Boolean) {
    isTrackingState = enabled
    statesTracker = if (isTrackingState) Some(new OptimizationStatesTracker()) else None
  }

  def getStateTracker: Option[OptimizationStatesTracker] = statesTracker

  def getInitialState: Option[OptimizerState] = initialState

  protected def setInitialState(state: Option[OptimizerState]): Unit = {
    initialState = state
  }

  def getCurrentState: Option[OptimizerState] = currentState

  protected def setCurrentState(state: Option[OptimizerState]): Unit = {
    currentState = state
    state match {
      case Some(x) =>
        statesTracker match {
          case Some(y) =>
            y.track(x)
            y.convergenceReason = convergenceReason
          case None =>
        }
      case None =>
    }
  }

  def getPreviousState: Option[OptimizerState] = previousState

  protected def setPreviousState(state: Option[OptimizerState]): Unit = {
    previousState = state
  }

  def stateTrackingEnabled: Boolean = isTrackingState

  protected def clearOptimizationStatesTracker(): Unit = {
    statesTracker match {
      case Some(x) => x.clear()
      case None =>
    }
  }
}
