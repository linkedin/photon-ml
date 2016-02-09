package com.linkedin.photon.ml.optimization

import breeze.linalg.Vector

/**
 * Similar to [[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.FirstOrderMinimizer\$State breeze.
 *   optimize.FirstOrderMinimizer.State]],
 * this class tracks the information about the optimizer, including the coefficients, the objective function's value and
 * gradient, and the current iteration number.
 * @param coefficients The current coefficients being optimized
 * @param value The current objective function's value
 * @param gradient The current objective function's gradient
 * @param iter what iteration number we are on
 * @author xazhang
 */
protected[optimization] case class OptimizerState(
  coefficients: Vector[Double],
  value: Double,
  gradient: Vector[Double],
  iter: Int)