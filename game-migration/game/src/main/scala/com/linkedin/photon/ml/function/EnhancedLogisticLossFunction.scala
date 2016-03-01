package com.linkedin.photon.ml.function

import breeze.linalg.{Vector, axpy}

import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Class for the logistic loss function:
 * sum_i (w_i*(y_i*log(1 + exp(-(theta'x_i + o_i))) + (1-y_i)*log(1 + exp(theta'x_i + o_i)))),
 * where \theta is the coefficients of the data features to be estimated, (y_i, x_i, o_i, w_i) are the tuple
 * for label, features, offset, and weight of the i'th labeled data point, respectively.
 * Note that the above equation assumes the label y_i \in {0, 1}, although the code below would also work when
 * y_i \in {-1, 1}.
 * @author xazhang
 */

class EnhancedLogisticLossFunction extends LogisticLossFunction with EnhancedTwiceDiffFunction[LabeledPoint] {
  override protected[ml] def hessianDiagonalAt(dataPoint: LabeledPoint,
                                                  coefficients: Vector[Double],
                                                  cumHessianDiagonal: Vector[Double]): Unit = {
    val LabeledPoint(_, features, _, weight) = dataPoint
    val margin = computeMargin(dataPoint, coefficients)
    val sigma = 1.0 / (1.0 + math.exp(-margin))
    val D = sigma * (1 - sigma)
    axpy(weight * D, features :* features, cumHessianDiagonal)
  }
}
