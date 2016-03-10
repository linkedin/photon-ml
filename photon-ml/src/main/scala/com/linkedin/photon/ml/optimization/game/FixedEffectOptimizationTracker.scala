package com.linkedin.photon.ml.optimization.game

import com.linkedin.photon.ml.optimization._

/**
 * Optimization tracker for factored randon effect optimization problems
 *
 * @param optimizationStatesTracker original state tracker for the optimization problem
 * @author xazhang
 */
class FixedEffectOptimizationTracker(optimizationStateTracker: OptimizationStatesTracker) extends OptimizationTracker {

  /**
   * Build a summary string for the tracker
   *
   * @return string representation
   */
  override def toSummaryString: String = optimizationStateTracker.toString()
}
