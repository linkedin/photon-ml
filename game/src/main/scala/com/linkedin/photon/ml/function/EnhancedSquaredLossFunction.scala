package com.linkedin.photon.ml.function

import breeze.linalg._

import com.linkedin.photon.ml.data.LabeledPoint


/**
 * @author xazhang
 */
class EnhancedSquaredLossFunction extends SquaredLossFunction with EnhancedTwiceDiffFunction[LabeledPoint] {
  override protected[ml] def hessianDiagonalAt(
      dataPoint: LabeledPoint,
      coefficients: Vector[Double],
      cumHessianDiagonal: Vector[Double]): Unit = {
    val LabeledPoint(_, features, _, weight) = dataPoint
    axpy(weight, features.map(feature => feature * feature), cumHessianDiagonal)
  }
}
