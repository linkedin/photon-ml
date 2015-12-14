package com.linkedin.photon.ml.optimization

import breeze.linalg.{Vector, norm}
import breeze.numerics.abs
import breeze.optimize.FirstOrderMinimizer._
import com.linkedin.photon.ml.data.DataPoint
import com.linkedin.photon.ml.function.DiffFunction
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

/**
 * Trait for optimization problem solvers.
 * @tparam Datum Generic type of input data point
 * @tparam Function Generic type of the objective function to be optimized.
 * @author xazhang
 * @author bdrew
 */
trait Optimizer[Datum <: DataPoint, -Function <: DiffFunction[Datum]] extends Serializable with Logging {

  /**
   * Maximum iterations
   */
  def getMaximumIterations(): Int

  /**
   * Set maximum iterations
   */
  def setMaximumIterations(maxIterations: Int): Unit

  /**
   * Map of feature index to bounds. Will be populated if the user specifies box constraints
   */
  def getConstraintMap: Option[Map[Int, (Double, Double)]]

  def setConstraintMap(constraintMap: Option[Map[Int, (Double, Double)]]): Unit

  /**
   * Initialize the context of the optimizer, e.g., the history of LBFGS and trust region size of Tron
   */
  protected def init(
    state: OptimizerState,
    data: Either[RDD[Datum], Iterable[Datum]],
    function: Function,
    coefficients: Vector[Double]): Unit

  /**
   * Clear the optimizer, e.g., the history of LBFGS and trust region size of Tron
   */
  def clear(): Unit

  /**
   * The initial state of the optimizer, used for checking convergence
   */
  def getInitialState(): Option[OptimizerState]

  /**
   * The current state of the optimizer
   */
  def getCurrentState: Option[OptimizerState]

  /**
   * The previous state of the optimizer
   */
  def getPreviousState: Option[OptimizerState]

  /**
   * Set the initial state for the optimizer
   * @param state
   */
  protected def setInitialState(state: Option[OptimizerState]): Unit

  /**
   * Set the current state for the optimizer
   *
   * @param state
   */
  protected def setCurrentState(state: Option[OptimizerState]): Unit

  /**
   * Set the previous state for the optimizer
   * @param state
   */
  protected def setPreviousState(state: Option[OptimizerState]): Unit

  /**
   * Get the convergence tolerance
   **/
  def getTolerance: Double

  /** Set the tolerance */
  def setTolerance(tolerance: Double): Unit

  /** Return our convergence reason */
  def convergenceReason: Option[ConvergenceReason]

  /** True if the optimizer thinks it's done. */
  def isDone: Boolean

  /** True if state tracking is enabled */
  def stateTrackingEnabled(): Boolean

  /** Set state tracking */
  def setStateTrackingEnabled(enabled: Boolean): Unit

  /** Get the state tracker */
  def getStateTracker: Option[OptimizationStatesTracker]

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
    clear()
    setCurrentState(Some(getState(data, objectiveFunction, initialCoefficients)))
    // initialize the optimizer state if it's not being initialized yet
    if (getInitialState.isEmpty) {
      setInitialState(getCurrentState)
    }
    init(getCurrentState.get, data, objectiveFunction, initialCoefficients)
    do {
      val updatedState = runOneIteration(data, objectiveFunction, getCurrentState.get)
      setPreviousState(getCurrentState)
      setCurrentState(Some(updatedState))
    } while (!isDone)

    val currentState = getCurrentState
    logInfo(s"After optimizing, current state: $currentState")
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
