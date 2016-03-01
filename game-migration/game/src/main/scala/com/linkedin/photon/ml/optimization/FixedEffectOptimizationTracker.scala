package com.linkedin.photon.ml.optimization

/**
 * @author xazhang
 */
class FixedEffectOptimizationTracker(optimizationStateTracker: OptimizationStatesTracker) extends OptimizationTracker {
  override def toSummaryString: String = optimizationStateTracker.toString()
}
