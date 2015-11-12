/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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
import breeze.optimize.FirstOrderMinimizer._
import com.linkedin.photon.ml.data.DataPoint
import com.linkedin.photon.ml.function.DiffFunction
import org.apache.spark.rdd.RDD

/**
 * Trait for optimization problem solvers.
 * @tparam Datum Generic type of input data point
 * @tparam Function Generic type of the objective function to be optimized.
 * @author xazhang
 */
trait Optimizer[Datum <: DataPoint, -Function <: DiffFunction[Datum]] extends Serializable {

  /**
   * The convergence tolerance of iterations for this optimizer. Smaller value will lead to higher
   * accuracy with the cost of more iterations (Tron and L-BFGS have different default tolerance parameter value).
   */
  var tolerance: Double = 1e-6

  /**
   * The maximal number of iterations for the optimizer (Tron and L-BFGS have different default maxNumIterations parameter value).
   */
  var maxNumIterations: Int = 80

  /**
   *  Map of feature index to bounds. Will be populated if the user specifies box constraints
   */
  var constraintMap: Option[Map[Int, (Double, Double)]] = None

  /**
   * Initialize the context of the optimizer, e.g., the history of LBFGS and trust region size of Tron
   */
  protected def init(state: OptimizerState, data: Either[RDD[Datum], Iterable[Datum]], function: Function, coefficients: Vector[Double]): Unit

  /**
   * Clean the context of the optimizer, e.g., the history of LBFGS and trust region size of Tron
   */
  protected def clean(): Unit

  /**
   * The initial state of the optimizer, used for checking convergence
   */
  private var initialState: Option[OptimizerState] = None

  /**
   * The current state of the optimizer
   */
  private var currentState: Option[OptimizerState] = None

  /**
   * The previous state of the optimizer
   */
  private var previousState: Option[OptimizerState] = None

  /**
   * Get the optimizer's state
   * @param data The training data
   * @param objectiveFunction The objective function to be optimized
   * @param coefficients The model coefficients
   */
  private def getState(data: Either[RDD[Datum], Iterable[Datum]],
                       objectiveFunction: Function,
                       coefficients: Vector[Double],
                       iter: Int = 0): OptimizerState = {
    val (value, gradient) = data match {
      //the calculation will be done in a distributed fashion
      case Left(dataAsRDD) =>
        val broadcastedCoefficients = dataAsRDD.context.broadcast(coefficients)
        val (value, gradient) = objectiveFunction.calculate(dataAsRDD, broadcastedCoefficients)
        broadcastedCoefficients.unpersist()
        (value, gradient)
      //the calculation will be done on a local machine.
      case Right(dataAsIterable) => objectiveFunction.calculate(dataAsIterable, coefficients)
    }
    OptimizerState(coefficients, value, gradient, iter)
  }

  /**
   * Clean the states of this optimizer
   */
  def cleanOptimizerState() = {
    clean()
    constraintMap = None
    initialState = None
    currentState = None
    previousState = None
  }

  def convergenceReason: Option[ConvergenceReason] = {
    if (initialState.isEmpty || currentState.isEmpty || previousState.isEmpty)
      None
    else if (currentState.get.iter >= maxNumIterations)
      Some(MaxIterations)
    else if (currentState.get.iter == previousState.get.iter)
      Some(ObjectiveNotImproving)
    else if (abs(currentState.get.value - previousState.get.value) <= tolerance * initialState.get.value)
      Some(FunctionValuesConverged)
    else if (norm(currentState.get.gradient, 2) <= tolerance * norm(initialState.get.gradient, 2))
      Some(GradientConverged)
    else
      None
  }

  /** True if the optimizer thinks it's done. */
  private def done = convergenceReason.nonEmpty

  /**
   * Whether to track the optimization states (for validating and debugging purpose)
   */
  protected[ml] var isTrackingState: Boolean = true

  /**
   * Track down the optimizer's states trajectory
   */
  protected var statesTracker: OptimizationStatesTracker = _

  protected[ml] def getStatesTracker: Option[OptimizationStatesTracker] = {
    if (isTrackingState) Some(statesTracker)
    else None
  }

  /**
   * Run one iteration of the optimizer given the current state
   * @param data The training data
   * @param objectiveFunction The objective function to be optimized
   * @param currentState The current optimizer state
   * @return The updated state of the optimizer
   */
  protected def runOneIteration(data: Either[RDD[Datum], Iterable[Datum]],
                                objectiveFunction: Function,
                                currentState: OptimizerState): OptimizerState

  /**
   * Solve the provided convex optimization problem.
   * @param data The training data
   * @param objectiveFunction The objective function to be optimized
   * @return Optimized coefficients and the optimized objective function's value
   */
  protected def optimize(data: Either[RDD[Datum], Iterable[Datum]],
                         objectiveFunction: Function): (Vector[Double], Double) = {
    val numFeatures =
      if (data.isLeft) {
        data.left.get.first().features.length
      } else {
        data.right.get.head.features.length
      }
    val initialCoefficients = Vector.zeros[Double](numFeatures)
    optimize(data, objectiveFunction, initialCoefficients)
  }

  /**
   * Solve the provided convex optimization problem.
   * @param data The training data
   * @param objectiveFunction The objective function to be optimized
   * @param initialCoefficients Initial coefficients
   * @return Optimized coefficients and the optimized objective function's value
   */
  protected def optimize(data: Either[RDD[Datum], Iterable[Datum]],
                         objectiveFunction: Function,
                         initialCoefficients: Vector[Double]): (Vector[Double], Double) = {
    if (isTrackingState) statesTracker = new OptimizationStatesTracker()
    currentState = Some(getState(data, objectiveFunction, initialCoefficients))
    // initialize the optimizer state if it's not being initialized yet
    if (initialState.isEmpty) {
      initialState = currentState
    }
    init(currentState.get, data, objectiveFunction, initialCoefficients)
    do {
      val updatedState = runOneIteration(data, objectiveFunction, currentState.get)
      previousState = currentState
      if (isTrackingState) statesTracker.track(updatedState)
      currentState = Some(updatedState)
    } while (!done)
    if (isTrackingState) statesTracker.convergenceReason = convergenceReason
    clean()
    (currentState.get.coefficients, currentState.get.value)
  }

  /**
   * Solve the provided convex optimization problem.
   * @param data The training data
   * @param objectiveFunction The objective function to be optimized
   * @return Optimized coefficients and the optimized objective function's value
   */
  def optimize(data: RDD[Datum], objectiveFunction: Function): (Vector[Double], Double) = {
    optimize(Left(data), objectiveFunction)
  }

  /**
   * Solve the provided convex optimization problem.
   * @param data The training data
   * @param objectiveFunction The objective function to be optimized
   * @param initialCoefficients Initial coefficients
   * @return Optimized coefficients and the optimized objective function's value
   */
  def optimize(data: RDD[Datum],
               objectiveFunction: Function,
               initialCoefficients: Vector[Double]): (Vector[Double], Double) = {
    optimize(Left(data), objectiveFunction, initialCoefficients)
  }

  /**
   * Solve the provided convex optimization problem.
   * @param data The training data
   * @param objectiveFunction The objective function to be optimized
   * @return Optimized coefficients and the optimized objective function's value
   */
  def optimize(data: Iterable[Datum], objectiveFunction: Function): (Vector[Double], Double) = {
    optimize(Right(data), objectiveFunction)
  }

  /**
   * Solve the provided convex optimization problem.
   * @param data The training data
   * @param objectiveFunction The objective function to be optimized
   * @param initialCoefficients Initial coefficients
   * @return Optimized coefficients and the optimized objective function's value
   */
  def optimize(data: Iterable[Datum],
               objectiveFunction: Function,
               initialCoefficients: Vector[Double]): (Vector[Double], Double) = {
    optimize(Right(data), objectiveFunction, initialCoefficients)
  }
}
