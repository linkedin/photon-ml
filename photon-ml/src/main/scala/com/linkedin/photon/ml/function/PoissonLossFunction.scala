package com.linkedin.photon.ml.function

import breeze.linalg.{Vector, axpy}
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Class for the Poisson loss function: sum_i (w_i*(exp(theta'x_i + o_i) - y_i*(theta'x_i + o_i))),
 * where \theta is the coefficients of the data features to be estimated, (y_i, x_i, o_i, w_i) are the tuple
 * for label, features, offset, and weight of the i'th labeled data point, respectively.
 * @author asaha
 */

class PoissonLossFunction extends TwiceDiffFunction[LabeledPoint] {
  /**
   * Calculate both the value and the gradient of the poisson loss function given one data point and the weight parameter,
   * with the computed gradient added to cumGradient
   * @param dataPoint The given labeled data points
   * @param coefficients The given model parameter
   * @param cumGradient The cumulative gradient
   * @return The computed value of the poisson loss function
   */
  override protected[ml] def calculateAt(dataPoint: LabeledPoint, coefficients: Vector[Double], cumGradient: Vector[Double]): Double = {
    val LabeledPoint(label, features, _, weight) = dataPoint
    val margin = computeMargin(dataPoint, coefficients)
    val expMargin = math.exp(margin)
    val gradientMultiplier = expMargin - label
    //val gradient = features*(weight*gradientMultiplier)
    /*
    use axpy to add the computed gradient to cumGradient in place, this is more efficient than first compute the gradient
    and then add it to cumGradient.
    */
    axpy(weight * gradientMultiplier, features, cumGradient)
    weight * (expMargin - label * margin)
  }

  /**
   * First calculate the Hessian of the poisson function under given one data point and coefficients,
   * then multiply it with a given vector and add to cumGradient
   * @param dataPoint The given data point
   * @param coefficients The given model parameter corresponding to the data features
   * @param vector The given vector to be multiplied with the Hessian
   * @param cumHessianVector The cumulative sum of the computed Hessian vector
   */
  override protected[ml] def hessianVectorAt(dataPoint: LabeledPoint,
                                                coefficients: Vector[Double],
                                                vector: Vector[Double],
                                                cumHessianVector: Vector[Double]): Unit = {
    val LabeledPoint(_, features, _, weight) = dataPoint
    val margin = computeMargin(dataPoint, coefficients)
    val expMargin = math.exp(margin)
    //val sigma = 1.0 / (1.0 + math.exp(-margin))
    //val D = sigma * (1 - sigma)
    axpy(weight * expMargin * features.dot(vector), features, cumHessianVector)
  }
}
