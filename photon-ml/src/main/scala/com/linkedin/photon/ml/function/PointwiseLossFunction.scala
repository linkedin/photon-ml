package com.linkedin.photon.ml.function

/**
 * The function to calculate l(z, y) with z = theta^T^x + offset for generalized linear model loss functions, where
 * theta is the coefficient, x is the feature vector, z is the margin, y is the label. It is the negative
 * of the log-likelihood for one data point, except for an irrelevant constant term.
 * For example, for linear regression, l(z, y) = 1/2 (z - y)^2^.
 */
trait PointwiseLossFunction extends Serializable {
  /**
   * Calculate the loss function value and 1st derivative with respect to z.
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The value and the 1st derivative
   */
  def loss(margin: Double, label: Double): (Double, Double)

  /**
   * Calculate the 2nd derivative with respect to z.
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The value and the 2st derivative with respect to z
   */
  def d2lossdz2(margin: Double, label: Double): Double
}
