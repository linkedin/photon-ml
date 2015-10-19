package com.linkedin.photon.ml.supervised

/**
 * Supported type of tasks
 * @author xazhang
 */
object TaskType extends Enumeration {
  type TaskType = Value
  val LINEAR_REGRESSION, POISSON_REGRESSION, LOGISTIC_REGRESSION = Value
}