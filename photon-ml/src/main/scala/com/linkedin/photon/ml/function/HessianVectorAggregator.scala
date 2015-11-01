package com.linkedin.photon.ml.function

import breeze.linalg.{Vector, axpy}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NormalizationContext


/**
 * An aggregator to perform calculation on Hessian vector multiplication for generalized linear model loss function, especially
 * in the context of normalization. Both iterable data and rdd data share the same logic for data aggregate.
 *
 * Refer to ***REMOVED*** for a better understanding
 * of the algorithm.
 *
 * Some logic of Hessian vector multiplication is the same for gradient aggregation, so this class inherits
 * ValueAndGradientAggregator.
 *
 * @param coef Coefficients (weights)
 * @param multiplyVector The vector to multiply with the Hessian matrix
 * @param func A single loss function for the generalized linear model
 * @param normalizationContext The normalization context
 *
 * @author dpeng
 */
@SerialVersionUID(2L)
protected[function] class HessianVectorAggregator(@transient coef: Vector[Double], @transient multiplyVector: Vector[Double],
                                                  func: PointwiseLossFunction, @transient normalizationContext: NormalizationContext) extends
     ValueAndGradientAggregator(coef, func, normalizationContext) {

  require(multiplyVector.size == dim)

  // effectiveMultiplyVector_j = factor_j * multiplyVector
  // This intermediate vector helps to facilitate calculating
  //    \sum_k (x_{ki} - shift_k) * factor_k * multiplyVector_k
  //  = \sum_k (x_{ki} - shift_k) * effectiveMultiplyVector_k
  // This vector is data point independent.
  val effectiveMultiplyVector: Vector[Double] = factorsOption match {
    case Some(factors) =>
      interceptIdOption.foreach(id =>
                                  require(factors(id) == 1.0,
                                          s"The intercept should not be transformed. Intercept " +
                                                  s"scaling factor: ${factors(id)}"))
      require(factors.size == dim, s"Size mismatch. Factors ")
      multiplyVector :* factors
    case None =>
      multiplyVector
  }

  // featureVectorProductShift = \sum_k shift_k * effectiveMultiplyVector_k
  // This intermediate value helps to facilitate calculating
  //     \sum_k (x_{ki} - shift_k) * factor_k * multiplyVector_k
  //   = \sum_k x_{ki} * effectiveMultiplyVector_k - featureVectorProductShift
  // This value is data point independent.
  val featureVectorProductShift: Double = shiftsOption match {
    case Some(shifts) =>
      effectiveMultiplyVector.dot(shifts)
    case None =>
      0.0
  }

  /**
   * Add a data point to the aggregator
   * @param datum a data point
   * @return The aggregator
   */
  override def add(datum: LabeledPoint): this.type = {
    totalCnt += 1
    val margin = datum.computeMargin(effectiveCoefficients) + marginShift
    val LabeledPoint(label, features, _, weight) = datum
    val d2ldz2 = func.d2lossdz2(margin, label)
    // l'' * (\sum_k x_{ki} * effectiveMultiplyVector_k - featureVectorProductShift)
    val effectiveWeight = weight * d2ldz2 * (features.dot(effectiveMultiplyVector) - featureVectorProductShift)

    vectorShiftPrefactorSum += effectiveWeight

    axpy(effectiveWeight, features, vectorSum)
    this
  }
}

