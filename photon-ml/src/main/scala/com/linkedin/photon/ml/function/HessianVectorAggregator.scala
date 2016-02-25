package com.linkedin.photon.ml.function

import breeze.linalg.{Vector, axpy}
import com.linkedin.photon.ml.data.{LabeledPoint, ObjectProvider}
import com.linkedin.photon.ml.normalization.NormalizationContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD


/**
 * An aggregator to perform calculation on Hessian vector multiplication for generalized linear model loss function,
 * especially in the context of normalization. Both iterable data and rdd data share the same logic for data aggregate.
 *
 * Refer to ***REMOVED*** for a better
 * understanding of the algorithm.
 *
 * Some logic of Hessian vector multiplication is the same for gradient aggregation, so this class inherits
 * ValueAndGradientAggregator.
 *
 * @param func A single loss function for the generalized linear model
 * @param dim The dimension (number of features) of the aggregator
 *
 * @author dpeng
 */
@SerialVersionUID(2L)
protected[ml] class HessianVectorAggregator(func: PointwiseLossFunction, dim: Int)
  extends ValueAndGradientAggregator(func, dim) {

  // effectiveMultiplyVector_j = factor_j * multiplyVector
  // This intermediate vector helps to facilitate calculating
  //    \sum_k (x_{ki} - shift_k) * factor_k * multiplyVector_k
  //  = \sum_k (x_{ki} - shift_k) * effectiveMultiplyVector_k
  // This vector is data point independent.
  @transient var effectiveMultiplyVector: Vector[Double] = _

  // featureVectorProductShift = \sum_k shift_k * effectiveMultiplyVector_k
  // This intermediate value helps to facilitate calculating
  //     \sum_k (x_{ki} - shift_k) * factor_k * multiplyVector_k
  //   = \sum_k x_{ki} * effectiveMultiplyVector_k - featureVectorProductShift
  // This value is data point independent.
  @transient var featureVectorProductShift: Double = _

  def init(
      datum: LabeledPoint,
      coef: Vector[Double],
      multiplyVector: Vector[Double],
      normalizationContext: NormalizationContext): Unit = {

    super.init(datum, coef, normalizationContext)
    require(multiplyVector.size == dim)
    val NormalizationContext(factorsOption, shiftsOption, interceptIdOption) = normalizationContext
    effectiveMultiplyVector = factorsOption match {
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
    featureVectorProductShift = shiftsOption match {
      case Some(shifts) =>
        effectiveMultiplyVector.dot(shifts)
      case None =>
        0.0
    }
  }

  /**
   * Add a data point to the aggregator
   * @param datum a data point
   * @return The aggregator
   */
  def add(
      datum: LabeledPoint,
      coef: Vector[Double],
      multiplyVector: Vector[Double],
      normalizationContext: NormalizationContext): this.type = {

    if (!initialized) {
      this.synchronized {
        init(datum, coef, multiplyVector, normalizationContext)
        initialized = true
      }
    }
    val LabeledPoint(label, features, _, weight) = datum
    require(features.size == dim, s"Size mismatch. Coefficient size: ${dim}, features size: ${features.size}")
    totalCnt += 1
    val margin = datum.computeMargin(effectiveCoefficients) + marginShift

    val d2ldz2 = func.d2lossdz2(margin, label)
    // l'' * (\sum_k x_{ki} * effectiveMultiplyVector_k - featureVectorProductShift)
    val effectiveWeight = weight * d2ldz2 * (features.dot(effectiveMultiplyVector) - featureVectorProductShift)

    vectorShiftPrefactorSum += effectiveWeight

    axpy(effectiveWeight, features, vectorSum)
    this
  }
}

object HessianVectorAggregator {
  def calcHessianVector(
      rdd: RDD[LabeledPoint],
      coef: Broadcast[Vector[Double]],
      multiplyVector: Broadcast[Vector[Double]],
      singleLossFunction: PointwiseLossFunction,
      normalizationContext: ObjectProvider[NormalizationContext],
      treeAggregateDepth: Int): Vector[Double] = {

    val aggregator = new HessianVectorAggregator(singleLossFunction, coef.value.size)

    val resultAggregator = rdd.treeAggregate(aggregator)(
      seqOp = (ag, datum) => ag.add(datum, coef.value, multiplyVector.value, normalizationContext.get),
      combOp = (ag1, ag2) => ag1.merge(ag2),
      depth = treeAggregateDepth
    )
    val result = resultAggregator.getVector(normalizationContext.get)
    result
  }

  def calcHessianVector(
      data: Iterable[LabeledPoint],
      coef: Vector[Double],
      multiplyVector: Vector[Double],
      singleLossFunction: PointwiseLossFunction,
      normalizationContext: ObjectProvider[NormalizationContext]): Vector[Double] = {

    val aggregator = new HessianVectorAggregator(singleLossFunction, coef.size)
    val resultAggregator = data.aggregate(aggregator)(
      seqop = (ag, datum) => ag.add(datum, coef, multiplyVector, normalizationContext.get),
      combop = (ag1, ag2) => ag1.merge(ag2)
    )
    resultAggregator.getVector(normalizationContext.get)
  }
}
