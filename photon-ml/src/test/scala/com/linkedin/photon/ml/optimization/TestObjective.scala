package com.linkedin.photon.ml.optimization

import breeze.linalg.{axpy, Vector, sum}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.TwiceDiffFunction

/**
 * Test function used solely to exercise the optimizers.
 * This function has known minimum at {@link TestFunction.CENTROID}.
 * @author bdrew
 */

class TestObjective extends TwiceDiffFunction[LabeledPoint] {

  override def calculateAt(dataPoint: LabeledPoint, parameter: Vector[Double], cumGradient: Vector[Double]): Double = {
    val delta = parameter - TestObjective.CENTROID
    val expDeltaSq = delta.mapValues { x => Math.exp(Math.pow(x, 2.0)) }
    cumGradient += expDeltaSq :* delta :* 2.0
    sum(expDeltaSq) - expDeltaSq.length
  }

  override def hessianVectorAt(dataPoint: LabeledPoint, parameter: Vector[Double],
                               vector: Vector[Double], cumHessianVector: Vector[Double]): Unit = {
    val delta = parameter - TestObjective.CENTROID
    val expDeltaSq = delta.mapValues { x => Math.exp(Math.pow(x, 2.0)) }
    val hess = expDeltaSq :* (delta :* delta :+ 1.0)
    cumHessianVector += hess :* vector :* 4.0
  }

  override protected[ml] def hessianDiagonalAt(
      dataPoint: LabeledPoint,
      coefficients: Vector[Double],
      cumHessianDiagonal: Vector[Double]): Unit = {

    val LabeledPoint(label, features, _, weight) = dataPoint
    axpy(weight, features.map(feature => feature * feature), cumHessianDiagonal)
  }

}

object TestObjective {
  val CENTROID: Double = Math.PI
}
