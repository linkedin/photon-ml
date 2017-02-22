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

import scala.collection.immutable.Queue

import breeze.linalg.norm

import com.linkedin.photon.ml.util.{ConvergenceReason, FunctionValuesConverged, GradientConverged, Logging}

/**
 * Class to track the history of an optimizer's states and wall-clock time elapsed per iteration.
 *
 * @param maxNumStates The maximum number of states to track. This is used to prevent the OptimizationHistoryTracker
 *                     from using too much memory to track the history of the states.
 * @note  DO NOT USE this class outside of Photon-ML. It is intended as an internal utility, and is likely to be
 *        changed or removed in future releases.
 */
protected[ml] class OptimizationStatesTracker(maxNumStates: Int = 100) extends Serializable with Logging {
  private val _startTime = System.currentTimeMillis()

  private var _times = Queue[Long]()
  private var _states = Queue[OptimizerState]()
  private var numStates = 0
  var convergenceReason: Option[ConvergenceReason] = None

  /**
   * Check if optimization ended because either the objective function value or gradient converged.
   *
   * @return True if either the function values or gradient converged, false otherwise
   */
  def converged: Boolean =
    convergenceReason.exists(_ == FunctionValuesConverged) || convergenceReason.exists(_ == GradientConverged)

  /**
   * Add the most recent state to the list of tracked states. If the limit of cached states is reached, remove the
   * oldest state.
   *
   * @param state The most recent state
   */
  def track(state: OptimizerState): Unit = {
    _times = _times.enqueue(System.currentTimeMillis() - _startTime)
    _states = _states.enqueue(state)
    numStates += 1
    if (numStates == maxNumStates) {
      _times = _times.dequeue._2
      _states = _states.dequeue._2
      numStates -= 1
    }
  }

  /**
   * Get the sequence of times between states as an Array.
   *
   * @return The times between states
   */
  def getTrackedTimeHistory: Array[Long] = _times.toArray

  /**
   * Get the sequence of recorded states as an Array.
   *
   * @return The recorded states
   */
  def getTrackedStates: Array[OptimizerState] = _states.toArray

  /**
   *
   * @return
   */
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
