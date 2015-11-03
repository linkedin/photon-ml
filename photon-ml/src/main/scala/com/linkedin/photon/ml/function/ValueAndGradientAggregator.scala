package com.linkedin.photon.ml.function

import breeze.linalg.{DenseVector, axpy, Vector}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NormalizationContext
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD


/**
 * An aggregator to perform calculation on value and gradient for generalized linear model loss function, especially
 * in the context of normalization. Both iterable data and rdd data share the same logic for data aggregate.
 *
 * Refer to ***REMOVED*** for a better understanding
 * of the algorithm.
 *
 * @param coef Coefficients (weights)
 * @param func A single loss function for the generalized linear model
 * @param normalizationContext The normalization context
 *
 * @author dpeng
 */
@SerialVersionUID(1L)
protected[function] class ValueAndGradientAggregator(@transient coef: Vector[Double], func: PointwiseLossFunction,
                                                     @transient normalizationContext: NormalizationContext) extends Serializable {
  // The transformation for a feature will be
  // x_i' = (x_i - shift_i) * factor_i
  @transient protected val NormalizationContext(factorsOption, shiftsOption, interceptIdOption) = normalizationContext
  protected val dim = coef.size

  // effectiveCoef = coef .* factor (point wise multiplication)
  // This is an intermediate vector to facilitate evaluation of value and gradient (and Hessian vector multiplication)
  // E.g., to calculate the margin:
  //     \sum_j coef_j * (x_j - shift_j) * factor_j
  //   = \sum_j coef_j * factor_j * x_j - \sum_j coef_j * factor_j * shift_j
  //   = \sum_j effectiveCoef_j * x_j - \sum_j effectiveCoef_j * shift_j
  //   = effectiveCoef^T x - effectiveCoef^T shift
  // This vector is data point independent.
  @transient protected val effectiveCoefficients = factorsOption match {
    case Some(factors) =>
      interceptIdOption.foreach(id =>
                                  require(factors(id) == 1.0, "The intercept should not be transformed. Intercept " +
                                          s"scaling factor: ${factors(id)}"))
      require(factors.size == dim, s"Size mismatch. Factors vector size: ${factors.size} != ${dim}.")
      coef :* factors
    case None =>
      coef
  }

  // Calculate: - effectiveCoef^T shift
  // This quantity is used to calculate the margin = effectiveCoef^T x - effectiveCoef^T shift
  // This value is datapoint independent.
  protected val marginShift = shiftsOption match {
    case Some(shifts) =>
      interceptIdOption.foreach(id =>
        require(shifts(id) == 0.0, s"The intercept should not be transformed. Intercept shift: ${shifts(shifts.length- 1)}"))
      - effectiveCoefficients.dot(shifts)
    case None =>
      0.0
  }

  // Total count
  protected var totalCnt = 0L

  // The accumulator to calculate the scaler.
  // For DiffFunction, this is \sum l(z_i, y_i) which sums up to objective value
  // For TwiceDiffFunction, this is not used
  protected var valueSum = 0.0d

  // The accumulator to calculate the principal part of the vector.
  // For DiffFunction, this is \sum l' x_{ji}, which sums up to the gradient without shifts and scaling
  //     gradient_j = \sum_i l'(z_i, y_i) (x_{ji} - shift_j) * factor_j
  //                = factor_j * [\sum_i l'(z_i, y_i) x_{ji} - shift_j * \sum_i l'(z_i, y_i)]
  //                = factor_j * [vectorSum - shift_j * \sum_i l'(z_i, y_i)]
  // For TwiceDiffFunction, this is \sum l''[(x-shift) * factor v] x, which sums up to the principal part of the Hessian vector product
  //     hv_j = \sum_ik (x_{ji} - shift_j) * factor_j * l''(z_i, y_i) * (x_{ki} - shift_k) * factor_k * v_k
  //          = \sum_i (x_{ji} - shift_j) * factor_j * l''(z_i, y_i) * \sum_k (x_{ki} - shift_k) * factor_k * v_k
  //          = factor_j * [\sum_i x_{ji} * l''(z_i, y_i) * \sum_k (x_{ki} - shift_k) * factor_k * v_k
  //                      - shift_j * \sum_i x_{ji} * l''(z_i, y_i) * \sum_k (x_{ki} - shift_k) * factor_k * v_k]
  //          = factor_j * [vectorSum - shift_j * \sum_i x_{ji} * l''(z_i, y_i) * \sum_k (x_{ki} - shift_k) * factor_k * v_k]
  protected val vectorSum: Vector[Double] = DenseVector.zeros[Double](dim)

  // The accumulator to calculate the prefactor of the vector shift.
  // For DiffFunction, this is \sum l', which sums up to the prefactor for gradient shift
  //      gradient_j = factor_j * [vectorSum - shift_j * vectorShiftPrefactorSum]
  // For TwiceDiffFuntion, this is \sum_i l''(z_i, y_i) * \sum_k (x_{ki} - shift_k) * factor_k * v_k,
  // which sums up to the prefactor for Hessian vector product shift
  //      hv_j = factor_j * (vectorSum - shift_j * vectorShiftPrefactorSum)
  protected var vectorShiftPrefactorSum = 0.0d

  protected var effectiveCoefBroadcast: Option[Broadcast[Vector[Double]]] = None

  /**
   * Initialize broadcast variables if RDD/SparkContext is used.
   * @param sc Spark context
   */
  def initBroadcast(sc: SparkContext): Unit = {
    effectiveCoefBroadcast = Some(sc.broadcast(effectiveCoefficients))
  }

  protected def cleanBroadcast(): Unit = {
    effectiveCoefBroadcast.foreach(x => x.destroy())
  }

  protected def getEffectiveCoef: Vector[Double] = if (effectiveCoefBroadcast.isDefined) {
    effectiveCoefBroadcast.get.value
  } else {
    effectiveCoefficients
  }

  /**
   * Add a data point
   * @param datum a data point
   * @return The aggregator
   */
  def add(datum: LabeledPoint): this.type = {
    val LabeledPoint(label, features, _, weight) = datum
    val localEffectiveCoef = getEffectiveCoef
    require(features.size == localEffectiveCoef.size, s"Size mismatch. Coefficient size: ${localEffectiveCoef.size}, features size: ${features.size}")
    totalCnt += 1
    val margin = datum.computeMargin(localEffectiveCoef) + marginShift

    val (l, dldz) = func.loss(margin, label)

    valueSum += weight * l
    vectorShiftPrefactorSum += weight * dldz
    axpy(weight * dldz, features, vectorSum)
    this
  }

  /**
   * Merge two aggregators
   * @param that The other aggregator
   * @return A merged aggregator
   */
  def merge(that: ValueAndGradientAggregator): this.type = {
    require(dim == that.dim, s"Dimension mismatch. this.dim=$dim, that.dim=${that.dim}")
    require(that.getClass.eq(getClass), s"Class mismatch. this.class=$getClass, that.class=${that.getClass}")
    if (that.totalCnt != 0) {
      totalCnt += that.totalCnt
      valueSum += that.valueSum
      vectorShiftPrefactorSum += that.vectorShiftPrefactorSum
      for (i <- 0 until dim) {
        vectorSum(i) += that.vectorSum(i)
      }
    }
    this
  }

  /**
   * Get the count
   * @return The  count
   */
  def count: Long = totalCnt

  /**
   * Return the objective value for ValueAndGradientAggregator. Not used in the HessianVectorAggregator
   * @return Return the objective value
   */
  def getValue: Double = valueSum

  /**
   * Return the gradient for ValueAndGradientAggregator, or the Hessian vector product for HessianVectorAggregator, especially
   * in the context of normalization.
   * @return Return the gradient for ValueAndGradientAggregator, or the Hessian vector product for HessianVectorAggregator
   */
  def getVector(normalizationContext: NormalizationContext): Vector[Double] = {
    val NormalizationContext(factorsOption, shiftsOption, _) = normalizationContext
    val result = DenseVector.zeros[Double](dim)
    (factorsOption, shiftsOption) match {
      case (Some(factors), Some(shifts)) =>
        for (i <- 0 until dim) {
          result(i) = (vectorSum(i) - shifts(i) * vectorShiftPrefactorSum) * factors(i)
        }
      case (Some(factors), None) =>
        for (i <- 0 until dim) {
          result(i) = vectorSum(i) * factors(i)
        }
      case (None, Some(shifts)) =>
        for (i <- 0 until dim) {
          result(i) = vectorSum(i) - shifts(i) * vectorShiftPrefactorSum
        }
      case (None, None) =>
        for (i <- 0 until dim) {
          result(i) = vectorSum(i)
        }
    }
    result
  }
}

object ValueAndGradientAggregator {
  def calculateValueAndGradient(rdd: RDD[LabeledPoint], coef: Vector[Double], singleLossFunction: PointwiseLossFunction, normalizationContext: NormalizationContext): (Double, Vector[Double]) = {
    val aggregator = new ValueAndGradientAggregator(coef, singleLossFunction,
                                                    normalizationContext)
    aggregator.initBroadcast(rdd.sparkContext)
    val resultAggregator = rdd.aggregate(aggregator)(
      seqOp = (ag, datum) => ag.add(datum),
      combOp = (ag1, ag2) => ag1.merge(ag2)
    )
    val result = (resultAggregator.getValue, resultAggregator.getVector(normalizationContext))
    resultAggregator.cleanBroadcast()
    result
  }

  def calculateValueAndGradient(data: Iterable[LabeledPoint], coef: Vector[Double], singleLossFunction: PointwiseLossFunction, normalizationContext: NormalizationContext): (Double, Vector[Double]) = {
    val aggregator = new ValueAndGradientAggregator(coef, singleLossFunction, normalizationContext)
    val resultAggregator = data.aggregate(aggregator)(
      seqop = (ag, datum) => ag.add(datum),
      combop = (ag1, ag2) => ag1.merge(ag2)
    )
    (resultAggregator.getValue, resultAggregator.getVector(normalizationContext))
  }
}
