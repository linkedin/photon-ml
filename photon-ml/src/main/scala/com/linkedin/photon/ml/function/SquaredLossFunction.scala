package com.linkedin.photon.ml.function

import breeze.linalg.{Vector, axpy}
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Class for the squared loss function: sum_i w_i/2*(theta'x_i + o_i - y_i)**2, where theta is the weight coefficients of
 * the data features to be estimated, (y_i, x_i, o_i, w_i) are the label, features, offset, and weight of
 * the i'th labeled data point, respectively.
 * @author xazhang
 */

class SquaredLossFunction extends TwiceDiffFunction[LabeledPoint] {

  override protected[ml] def calculateAt(dataPoint: LabeledPoint, coefficients: Vector[Double], cumGradient: Vector[Double]): Double = {
    val LabeledPoint(label, features, _, weight) = dataPoint
    val margin = computeMargin(dataPoint, coefficients)
    val diff = margin - label
    //calculating the gradient of the squared loss function and add it in-place to cumGradient
    axpy(weight * diff, features, cumGradient)
    weight * diff * diff / 2.0
  }

  override protected[ml] def hessianVectorAt(dataPoint: LabeledPoint,
                                                coefficients: Vector[Double],
                                                multiplyVector: Vector[Double],
                                                cumHessianVector: Vector[Double]): Unit = {
    val LabeledPoint(_, features, _, weight) = dataPoint
    //calculating the hessian multiplyVector of the squared loss function and add it in-place to cumHessianVector
    axpy(weight * features.dot(multiplyVector), features, cumHessianVector)
  }
}