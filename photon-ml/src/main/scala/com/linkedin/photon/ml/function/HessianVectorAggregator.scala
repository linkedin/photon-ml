package com.linkedin.photon.ml.function

import breeze.linalg.{Vector, axpy}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NormalizationContext
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD


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
  @transient val effectiveMultiplyVector: Vector[Double] = factorsOption match {
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

  var effectiveMultiplyVectorBroadcast: Option[Broadcast[Vector[Double]]] = None

  override def initBroadcast(sc: SparkContext): Unit = {
    super.initBroadcast(sc)
    effectiveMultiplyVectorBroadcast = Some(sc.broadcast(effectiveMultiplyVector))
  }

  override def cleanBroadcast(): Unit = {
    super.cleanBroadcast()
    effectiveMultiplyVectorBroadcast.foreach(x => x.destroy())
  }

  protected def getEffectiveMultiplyVector: Vector[Double] = effectiveMultiplyVectorBroadcast match {
    case Some(broadcast) =>
      broadcast.value
    case None =>
      effectiveMultiplyVector
  }

  /**
   * Add a data point to the aggregator
   * @param datum a data point
   * @return The aggregator
   */
  override def add(datum: LabeledPoint): this.type = {
    val localEffectiveCoef = getEffectiveCoef
    val LabeledPoint(label, features, _, weight) = datum
    require(features.size == localEffectiveCoef.size, s"Size mismatch. Coefficient size: ${localEffectiveCoef.size}, features size: ${features.size}")
    totalCnt += 1
    val margin = datum.computeMargin(localEffectiveCoef) + marginShift

    val d2ldz2 = func.d2lossdz2(margin, label)
    // l'' * (\sum_k x_{ki} * effectiveMultiplyVector_k - featureVectorProductShift)
    val effectiveWeight = weight * d2ldz2 * (features.dot(getEffectiveMultiplyVector) - featureVectorProductShift)

    vectorShiftPrefactorSum += effectiveWeight

    axpy(effectiveWeight, features, vectorSum)
    this
  }
}

object HessianVectorAggregator {
  def calcHessianVector(rdd: RDD[LabeledPoint], coef: Vector[Double], multiplyVector: Vector[Double],
                        singleLossFunction: PointwiseLossFunction, normalizationContext: NormalizationContext): Vector[Double] = {
    val aggregator = new HessianVectorAggregator(coef, multiplyVector,
                                                 singleLossFunction, normalizationContext)
    aggregator.initBroadcast(rdd.sparkContext)
    val resultAggregator = rdd.aggregate(aggregator)(
      seqOp = (ag, datum) => ag.add(datum),
      combOp = (ag1, ag2) => ag1.merge(ag2)
    )
    val result = resultAggregator.getVector(normalizationContext)
    aggregator.cleanBroadcast()
    result
  }

  def calcHessianVector(data: Iterable[LabeledPoint], coef: Vector[Double], multiplyVector: Vector[Double],
                        singleLossFunction: PointwiseLossFunction, normalizationContext: NormalizationContext): Vector[Double] = {
    val aggregator = new HessianVectorAggregator(coef, multiplyVector,
                                                 singleLossFunction, normalizationContext)
    val resultAggregator = data.aggregate(aggregator)(
      seqop = (ag, datum) => ag.add(datum),
      combop = (ag1, ag2) => ag1.merge(ag2)
    )
    resultAggregator.getVector(normalizationContext)
  }
}
