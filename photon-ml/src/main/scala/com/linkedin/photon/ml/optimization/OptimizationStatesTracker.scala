package com.linkedin.photon.ml.optimization

import breeze.linalg.norm
import breeze.optimize.FirstOrderMinimizer.{ConvergenceReason, FunctionValuesConverged, GradientConverged}

import scala.collection.mutable

/**
 * Class to track the history of an optimizer's states and wall-clock time elapsed per iteration
 * @param maxNumStates The maximum number of states to track. This is used to prevent the OptimizationHistoryTracker
 *                     from using too much memory to track the history of the states.
 * @author xazhang
 * @note  DO NOT USE this class outside of Photon-ML. It is intended as an internal utility, and is likely to be changed or removed in future releases.
 */
protected[ml] class OptimizationStatesTracker(maxNumStates: Int = 100) extends Serializable {

  private val _times = new mutable.ArrayBuffer[Long]
  private val _states = new mutable.ArrayBuffer[OptimizerState]
  private val _startTime = System.currentTimeMillis()

  var convergenceReason: Option[ConvergenceReason] = None

  /** True if the optimizer is done because either function values converged or gradient converged  */
  def converged = convergenceReason == Some(FunctionValuesConverged) || convergenceReason == Some(GradientConverged)

  private var numStates = 0

  def clear() = {
    _times.clear()
    _states.clear()
  }

  def track(state: OptimizerState): Unit = {
    _times += System.currentTimeMillis() - _startTime
    _states += state
    numStates += 1
    if (numStates == maxNumStates) {
      _times.remove(0)
      _states.remove(0)
      numStates -= 1
    }
  }

  def getTrackedTimeHistory: Array[Long] = _times.toArray

  def getTrackedStates: Array[OptimizerState] = _states.toArray

  override def toString: String = {
    val stringBuilder = new StringBuilder
    val convergenceReasonStr = convergenceReason match {
      case Some(reason) => reason.reason
      case None => "Optimizer is not converged properly, please check the log for more information"
    }
    val timeElapsed = getTrackedTimeHistory
    val states = getTrackedStates
    stringBuilder ++= s"Convergence reason: $convergenceReasonStr\n"
    val strIter = "Iter"
    val strTime = "Time(s)"
    val strValue = "Value"
    val strGradient = "|Gradient|"
    stringBuilder ++= f"$strIter%10s$strTime%10s$strValue%25s$strGradient%15s\n"
    stringBuilder ++= states.zip(timeElapsed).map { case (OptimizerState(_, value, gradient, iter), time) =>
      f"$iter%10d${time * 0.001}%10.3f$value%25.8f${norm(gradient, 2)}%15.2e"
    }.mkString("\n")
    stringBuilder ++= "\n"
    stringBuilder.result()
  }
}
