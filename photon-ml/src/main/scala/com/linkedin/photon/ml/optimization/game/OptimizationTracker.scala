package com.linkedin.photon.ml.optimization.game

/**
 * Optimization tracker
 *
 * @author xazhang
 */
trait OptimizationTracker {

  /**
   * Build a summary string for the tracker
   *
   * @return string representation
   */
  def toSummaryString: String
}
