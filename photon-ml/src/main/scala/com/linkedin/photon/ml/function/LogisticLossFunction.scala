package com.linkedin.photon.ml.function

import breeze.linalg.{Vector, axpy}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.util.Utils

/**
 * Class for the logistic loss function: sum_i (w_i*(y_i*log(1 + exp(-(theta'x_i + o_i))) + (1-y_i)*log(1 + exp(theta'x_i + o_i)))),
 * where \theta is the coefficients of the data features to be estimated, (y_i, x_i, o_i, w_i) are the tuple
 * for label, features, offset, and weight of the i'th labeled data point, respectively.
 * Note that the above equation assumes the label y_i \in {0, 1}. However, the code below would also work when y_i \in {-1, 1}.
 * @author xazhang
 */

class LogisticLossFunction extends TwiceDiffFunction[LabeledPoint] {

  override protected[ml] def calculateAt(dataPoint: LabeledPoint, coefficients: Vector[Double], cumGradient: Vector[Double]): Double = {
    val LabeledPoint(label, features, _, weight) = dataPoint
    val margin = computeMargin(dataPoint, coefficients)
    /*
    use axpy to add the computed gradient to cumGradient in place, this is more efficient than first compute the gradient
    explicitly and then add it to cumGradient.
     */
    if (label > 0) {
      axpy(weight * (1.0 / (1.0 + math.exp(-margin)) - 1.0), features, cumGradient)
      // The following is equivalent to log(1 + exp(-margin)) but more numerically stable.
      weight * Utils.log1pExp(-margin)
    } else {
      axpy(weight * (1.0 - 1.0 / (1.0 + math.exp(margin))), features, cumGradient)
      weight * Utils.log1pExp(margin)
    }
  }

  override protected[ml] def hessianVectorAt(dataPoint: LabeledPoint,
                                                coefficients: Vector[Double],
                                                vector: Vector[Double],
                                                cumHessianVector: Vector[Double]): Unit = {
    val LabeledPoint(_, features, _, weight) = dataPoint
    val margin = computeMargin(dataPoint, coefficients)
    val sigma = 1.0 / (1.0 + math.exp(-margin))
    val D = sigma * (1 - sigma)
    axpy(weight * D * features.dot(vector), features, cumHessianVector)
  }
}