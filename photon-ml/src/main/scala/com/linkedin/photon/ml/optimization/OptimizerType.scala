package com.linkedin.photon.ml.optimization

/**
 * Supported optimizers types
 * @author xazhang
 */
object OptimizerType extends Enumeration {
  type OptimizerType = Value
  val LBFGS, TRON = Value
}
