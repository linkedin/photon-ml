package com.linkedin.photon.ml.optimization.game

import com.linkedin.photon.ml.optimization._

/**
 * @author xazhang
 */
class FixedEffectOptimizationTracker(optimizationStateTracker: OptimizationStatesTracker) extends OptimizationTracker {
  override def toSummaryString: String = optimizationStateTracker.toString()
}
