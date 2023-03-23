/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.aggregators

import breeze.linalg.{DenseVector, Vector, axpy}
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.glm.PointwiseLossFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.{BroadcastWrapper, PhotonBroadcast, PhotonNonBroadcast}

/**
 * An aggregator to calculate the value and gradient of a generalized linear model loss function (possibly with
 * normalization) on a dataset.
 *
 * @param lossFunction The GLM loss function
 * @param coefficients The coefficients for the GLM
 * @param normalizationContext The feature normalization information
 */
@SerialVersionUID(2L)
protected[ml] class ValueAndGradientAggregator(
    lossFunction: PointwiseLossFunction,
    coefficients: BroadcastWrapper[Vector[Double]],
    normalizationContext: BroadcastWrapper[NormalizationContext]) extends Serializable {

  checkInvariants()

  /**
   * The equation for measuring the loss for a particular set of coefficients over a dataset is as follows:
   *
   * L(w, X, y, u, o) = \sum_i u_i * l((X_i * w) + o_i, y_i)
   *                  = \sum_i u_i * l(z_i, y_i)
   *
   * z_i = (X_i * w) + o_i
   *
   * where:
   *
   * w = The vector of coefficients, dimension m
   * X = The matrix of input data, dimension n x m
   * y = The vector of labels, dimension n
   * u = The vector of weights, dimension n
   * o = The vector of offsets (prior/residual values), dimension n
   * l(z, y) = The loss function for the particular optimization problem. For example, for linear regression:
   *
   *           l(z, y) = ((μ(z) - y)^2^) / 2
   *
   *           where μ(z) if the mean function for the class of GLM (in this case, μ(z) = z)
   *
   *
   * When the features are normalized, the above equation is modified:
   *
   * z_i = (((X_i -:- s) *:* f) * w) + o_i
   *
   * where:
   *
   * s = The vector of shifts for the features
   * f = The vector of scaling factors for the features
   *
   *
   * If we expand this equation, we see that it contains two terms that are identical for all i:
   *
   * z_i = (((X_i -:- s) *:* f) * w) + o_i
   *     = (((X_i *:* f) -:- (s *:* f)) * w) + o_i
   *     = ((X_i *:* f) * w) - ((s *:* f) * w) + o_i
   *     = (X_i * (f *:* w)) - (s * (f *:* w)) + o_i
   *     = (X_i * ew) - es + o_i
   *
   * ew = f *:* w
   *
   * es = s * ew
   *
   * where:
   *
   * ew = The vector of effective coefficients
   * es = The total shift
   */

  /**
   * Corresponds to the ew vector in the above equations
   */
  @transient lazy protected val effectiveCoefficients: Vector[Double] = normalizationContext.value.factorsOpt match {
    case Some(factors) => coefficients.value *:* factors
    case None => coefficients.value
  }

  /**
   * Corresponds to the es value in the above equations
   */
  @transient lazy protected val totalShift: Double = normalizationContext.value.shiftsAndInterceptOpt match {
    case Some((shifts, _)) => -1 * effectiveCoefficients.dot(shifts)
    case None => 0D
  }

  /**
   * Counter for number of aggregated data points
   */
  protected var totalCnt = 0L

  /**
   * Aggregator variable for total loss
   */
  protected var valueSum = 0.0d

  /**
   * Expanding from the above, the equation for computing the gradient for a particular set of coefficients over a
   * dataset is as follows:
   *
   * G_j = ∂ L(w, X, y, u, o) / ∂ w_j
   *     = \sum_i ∂ (u_i * l(z_i, y_i)) / ∂ w_j
   *     = \sum_i (∂ l(z_i, y_i) / ∂ z_i) * u_i * X_i_j
   *
   * Therefore:
   *
   * G = \sum_i e((∂ l(z_i, y_i) / ∂ z_i) * u_i, m) *:* X_i
   *
   * where:
   *
   * e(a, b) = A vector of length b where each value is a
   *
   *
   * When the features are normalized, the above equation is modified:
   *
   * G = \sum_i e((∂ l(z_i, y_i) / ∂ z_i) * u_i, m) *:* (X_i -:- s) *:* f
   *   = f *:* \sum_i e((∂ l(z_i, y_i) / ∂ z_i) * u_i, m) *:* (X_i -:- s)
   *   = f *:* ((\sum_i e((∂ l(z_i, y_i) / ∂ z_i) * u_i, m) *:* X_i) -:- (\sum_i e((∂ l(z_i, y_i) / ∂ z_i) * u_i, m) *:* s))
   *
   * For performance reasons, during aggregation we keep track of two values:
   *
   * v = \sum_i e((∂ l(z_i, y_i) / ∂ z_i) * u_i, m) *:* X_i
   * sf = \sum_i (∂ l(z_i, y_i) / ∂ z_i) * u_i
   *
   * Which results in the final equation:
   *
   * G = f *:* (v -:- (e(sf, m) *:* s))
   */

  /**
   * Aggregator value for the v vector in the above equations
   */
  protected var vectorSum: Vector[Double] = DenseVector.zeros[Double](coefficients.value.length)

  /**
   * Aggregator value for the sf vector in the above equations
   */
  protected var shiftFactor = 0.0d

  /**
   * Invariants that must hold for every instance of [[ValueAndGradientAggregator]].
   */
  protected def checkInvariants(): Unit = {

    normalizationContext.value match {

      case NormalizationContext(Some(factors), Some((shifts, _))) =>

        val coefficientsLength = coefficients.value.length

        require(
          coefficientsLength == factors.length,
          s"Length mismatch between coefficients and normalization factors: " +
            s"coefficients = $coefficientsLength, factors = ${factors.length}")
        require(
          coefficientsLength == shifts.length,
          s"Length mismatch between coefficients and normalization shifts: " +
            s"coefficients = $coefficientsLength, shifts = ${shifts.length}")

      case NormalizationContext(Some(factors), None) =>

        require(
          coefficients.value.length == factors.length,
          s"Length mismatch between coefficients and normalization factors: " +
            s"coefficients = ${coefficients.value.length}, factors = ${factors.length}")

      case NormalizationContext(None, Some((shifts, _))) =>

        require(
          coefficients.value.length == shifts.length,
          s"Length mismatch between coefficients and normalization shifts: " +
            s"coefficients = ${coefficients.value.length}, shifts = ${shifts.length}")

      case NormalizationContext(None, None) =>
    }
  }

  /**
   * Getter for the count of aggregated points.
   *
   * @return The count of aggregated points
   */
  def count: Long = totalCnt

  /**
   * Getter for the objective value.
   *
   * @return The objective value
   */
  def value: Double = valueSum

  /**
   * Getter for the cumulative gradient.
   *
   * @return The cumulative gradient
   */
  def gradient: Vector[Double] = {

    val NormalizationContext(factorsOpt, shiftsAndInterceptOpt) = normalizationContext.value

    (factorsOpt, shiftsAndInterceptOpt) match {
      case (Some(factors), Some((shifts, _))) =>
        (vectorSum -:- (shifts * shiftFactor)) *:* factors

      case (Some(factors), None) =>
        vectorSum *:* factors

      case (None, Some((shifts, _))) =>
        vectorSum -:- (shifts * shiftFactor)

      case (None, None) =>
        vectorSum
    }
  }

  /**
   * Add a data point to the aggregator.
   *
   * @param datum The data point
   * @return This aggregator object itself
   */
  def add(datum: LabeledPoint): this.type = {

    val LabeledPoint(label, features, _, weight) = datum

    require(
      features.size == effectiveCoefficients.size,
      s"Size mismatch: coefficients size = ${effectiveCoefficients.size}, features size = ${features.size}")

    val margin = datum.computeMargin(effectiveCoefficients) + totalShift
    val (loss, dzLoss) = lossFunction.lossAndDzLoss(margin, label)

    totalCnt += 1
    valueSum += weight * loss
    shiftFactor += weight * dzLoss
    axpy(weight * dzLoss, features, vectorSum)

    this
  }

  /**
   * Merge two aggregators.
   *
   * @param that The other aggregator
   * @return This aggregator object itself, with the contents of the other aggregator object merged into it
   */
  def merge(that: ValueAndGradientAggregator): this.type = {

    // TODO: The below tests are technically correct, but unnecessarily slow. Currently the only time that two
    // TODO: aggregators are merged is during aggregation, where they are copies of the same initial object.
//    require(lossFunction.equals(that.lossFunction), "Attempting to merge aggregators with different loss functions")
//    require(coefficients == that.coefficients, "Attempting to merge aggregators with different coefficients vectors")
//    require(
//      normalizationContext == that.normalizationContext,
//      "Attempting to merge aggregators with different normalization")

    if (that.count != 0) {
      totalCnt += that.totalCnt
      valueSum += that.valueSum
      shiftFactor += that.shiftFactor
      vectorSum :+= that.vectorSum
    }

    this
  }
}

object ValueAndGradientAggregator {

  /**
   * Calculate the value and gradient for an objective function in Spark.
   *
   * @param input An RDD of data points
   * @param coefficients The current model coefficients
   * @param singleLossFunction The function used to compute loss for predictions
   * @param normalizationContext The normalization context
   * @param treeAggregateDepth The tree aggregate depth
   * @return A tuple of the value and gradient
   */
  def calculateValueAndGradient(
      input: RDD[LabeledPoint],
      coefficients: Vector[Double],
      singleLossFunction: PointwiseLossFunction,
      normalizationContext: BroadcastWrapper[NormalizationContext],
      treeAggregateDepth: Int): (Double, Vector[Double]) = {

    val coefficientsBroadcast = input.sparkContext.broadcast(coefficients)
    val aggregator = new ValueAndGradientAggregator(
      singleLossFunction,
      PhotonBroadcast(coefficientsBroadcast),
      normalizationContext)
    val resultAggregator = input.treeAggregate(
      aggregator)(
      seqOp = (ag, datum) => ag.add(datum),
      combOp = (ag1, ag2) => ag1.merge(ag2),
      depth = treeAggregateDepth)

    coefficientsBroadcast.unpersist()

    (resultAggregator.value, resultAggregator.gradient)
  }

  /**
   * Calculate the value and gradient for an objective function locally.
   *
   * @param input An iterable set of data points
   * @param coefficients The current model coefficients
   * @param singleLossFunction The function used to compute loss for predictions
   * @param normalizationContext The normalization context
   * @return A tuple of the value and gradient
   */
  def calculateValueAndGradient(
      input: Iterable[LabeledPoint],
      coefficients: Vector[Double],
      singleLossFunction: PointwiseLossFunction,
      normalizationContext: BroadcastWrapper[NormalizationContext]): (Double, Vector[Double]) = {

    val aggregator = new ValueAndGradientAggregator(
      singleLossFunction,
      PhotonNonBroadcast(coefficients),
      normalizationContext)
    val resultAggregator = input.aggregate(
      aggregator)(
      seqop = (ag, datum) => ag.add(datum),
      combop = (ag1, ag2) => ag1.merge(ag2))

    (resultAggregator.value, resultAggregator.gradient)
  }
}
