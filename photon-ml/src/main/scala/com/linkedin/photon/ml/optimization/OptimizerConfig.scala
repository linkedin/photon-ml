package com.linkedin.photon.ml.optimization

import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType

/**
 * Contains configuration information for Optimizer instances.
 */
case class OptimizerConfig(
  optimizerType: OptimizerType,
  maximumIterations: Int,
  tolerance: Double,
  constraintMap: Option[Map[Int, (Double, Double)]])
