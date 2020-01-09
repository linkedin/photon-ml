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
package com.linkedin.photon.ml.function.glm

import breeze.linalg.{Vector, axpy}
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.{BroadcastWrapper, PhotonBroadcast, PhotonNonBroadcast}

/**
 * An aggregator to calculate the value of a multiplication between the Hessian matrix of a generalized linear model
 * loss function (possibly with normalization) on a dataset and some vector d.
 *
 * This class is implemented as a convenience to more efficiently compute this value, rather than computing the entire
 * Hessian matrix and performing a multiplication operation with vector d afterwards.
 *
 * @param lossFunction The GLM loss function
 * @param coefficients The coefficients for the GLM
 * @param dVector The vector d to multiply with the Hessian matrix
 * @param normalizationContext The feature normalization information
 */
@SerialVersionUID(3L)
protected[ml] class HessianVectorAggregator(
    lossFunction: PointwiseLossFunction,
    coefficients: BroadcastWrapper[Vector[Double]],
    dVector: BroadcastWrapper[Vector[Double]],
    normalizationContext: BroadcastWrapper[NormalizationContext])
  extends ValueAndGradientAggregator(lossFunction, coefficients, normalizationContext) {

  /**
   * Expanding upon the documentation in the [[ValueAndGradientAggregator]], the equation for computing the Hessian
   * matrix for a particular set of coefficients over a dataset is as follows:
   *
   * H_jk = ∂^2^ L(w, X, y, u, o) / (∂ w_j)(∂ w_k)
   *      = \sum_i ∂^2^ (u_i * l(z_i, y_i)) / (∂ w_j)(∂ w_k)
   *      = \sum_i (∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * X_i_j * X_i_k
   *
   * Let v = H * d, where d is some vector. Then v_j = H_j * d, where:
   *
   * H_j = \sum_i H_i_j         =>
   *
   * v_j = (\sum_i H_i_j) * d
   *     = \sum_i H_i_j * d
   *     = \sum_i v_i_j
   *
   * v_i_j = H_i_j * d          =>
   *
   * v_i = H_i * d
   *
   *
   * Returning to the above Hessian equation:
   *
   * H_jk = \sum_i (∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * X_i_j * X_i_k              =>
   *
   * H_i_jk = (∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * X_i_j * X_i_k                   =>
   *
   * v_i_j = \sum_k H_i_jk * d_k
   *       = \sum_k (∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * X_i_j * X_i_k * d_k
   *       = (∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * X_i_j * (\sum_k X_i_k * d_k)
   *       = (∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * X_i_j * (X_i * d)
   *       = (∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * (X_i * d) * X_i_j
   *
   * Therefore:
   *
   * v_i = e((∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * (X_i * d), m) *:* X_i
   *
   * v = \sum_i e((∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * (X_i * d), m) *:* X_i
   *
   *
   * When the features are normalized, the above equation is modified:
   *
   * v = \sum_i e((∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * (((X_i -:- s) *:* f) * d), m) *:* ((X_i -:- s) *:* f)
   *     \sum_i e((∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * ((X_i -:- s) * (f *:* d)), m) *:* ((X_i -:- s) *:* f)
   *     \sum_i e((∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * ((X_i * (f *:* d)) - (s * (f *:* d))), m) *:* ((X_i -:- s) *:* f)
   *
   * In the above equation, there are two terms that are identical for all i:
   *
   * ed = f *:* d
   *
   * eds = s * ed
   *
   * where:
   *
   * ed = The effective multiplication vector
   * eds = The total shift of the multiplication vector
   *
   * In addition, let us also define:
   *
   * g_i = e((∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * ((X_i * ed) - eds), m)
   *
   * Therefore:
   *
   * v = \sum_i g_i *:* ((X_i -:- s) *:* f)
   *   = \sum_i g_i *:* ((X_i *:* f) -:- (s *:* f))
   *   = \sum_i (g_i *:* X_i *:* f) -:- (g_i *:* s *:* f)
   *   = f *:* (\sum_i (g_i *:* X_i) -:- (g_i *:* s))
   *   = f *:* ((\sum_i g_i *:* X_i) -:- (\sum_i g_i *:* s))
   *   = f *:* ((\sum_i g_i *:* X_i) -:- (s *:* \sum_i g_i))
   */

  /**
   * Corresponds to the ed vector in the above equations
   */
  @transient lazy private val effectiveMultiplyVector: Vector[Double] = normalizationContext.value.factorsOpt match {
    case Some(factors) => dVector.value *:* factors
    case None => dVector.value
  }

  /**
   * Corresponds to the eds value in the above equations
   */
  @transient lazy private val featureVectorProductShift: Double = normalizationContext.value.shiftsAndInterceptOpt match {
    case Some((shifts, _)) => effectiveMultiplyVector.dot(shifts)
    case None => 0.0
  }

  /**
   * Invariants that must hold for every instance of [[HessianVectorAggregator]].
   */
  override protected def checkInvariants(): Unit = {

    super.checkInvariants()

    require(
      dVector.value.length == coefficients.value.length,
      s"Length mismatch between coefficients and d vector:" +
        s"coefficients = ${coefficients.value.length}, d = ${dVector.value.length}")
  }

  /**
   * Getter for the objective value. Not used in [[HessianVectorAggregator]].
   *
   * @return The objective value
   */
  override def value: Double =
    throw new IllegalAccessException(s"Function 'value' called for ${this.getClass.getSimpleName}")

  /**
   * Getter for the cumulative gradient. Not used in [[HessianVectorAggregator]].
   *
   * @return The cumulative gradient
   */
  override def gradient: Vector[Double] =
    throw new IllegalAccessException(s"Function 'gradient' called for ${this.getClass.getSimpleName}")

  /**
   * Getter for the multiplication between the Hessian matrix and the vector d.
   *
   * The vectorSum and shiftFactor members of the [[ValueAndGradientAggregator]] are reused. Using the terminology of
   * the documentation above:
   *
   * vectorSum = \sum_i g_i *:* X_i
   * shiftFactor = \sum_i g_i
   *
   * @return The multiplication between the Hessian matrix and the vector d
   */
  def hessianD: Vector[Double] = {

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
   * The vectorSum and shiftFactor members of the [[ValueAndGradientAggregator]] are reused. Using the terminology of
   * the documentation above:
   *
   * vectorSum = \sum_i g_i *:* X_i
   * shiftFactor = \sum_i g_i
   *
   * @param datum The data point
   * @return This aggregator object itself
   */
  override def add(datum: LabeledPoint): this.type = {

    val LabeledPoint(label, features, _, weight) = datum

    require(
      features.size == effectiveCoefficients.size,
      s"Size mismatch: coefficients size = ${effectiveCoefficients.size}, features size = ${features.size}")

    val margin = datum.computeMargin(effectiveCoefficients) + totalShift
    val dzzLoss = lossFunction.DzzLoss(margin, label)
    val effectiveWeight = weight * dzzLoss * (features.dot(effectiveMultiplyVector) - featureVectorProductShift)

    totalCnt += 1
    shiftFactor += effectiveWeight
    axpy(effectiveWeight, features, vectorSum)

    this
  }
}

object HessianVectorAggregator {

  /**
   * Calculate the multiplication between the Hessian matrix for an objective function and some vector d in Spark.
   *
   * @param input An RDD of data points
   * @param coefficients The current model coefficients
   * @param dVector The Hessian multiplication vector
   * @param singleLossFunction The function used to compute loss for predictions
   * @param normalizationContext The normalization context
   * @param treeAggregateDepth The tree aggregate depth
   * @return The multiplication between the Hessian matrix and some vector d
   */
  def calcHessianVector(
      input: RDD[LabeledPoint],
      coefficients: Vector[Double],
      dVector: Vector[Double],
      singleLossFunction: PointwiseLossFunction,
      normalizationContext: BroadcastWrapper[NormalizationContext],
      treeAggregateDepth: Int): Vector[Double] = {

    val coefficientsBroadcast = input.sparkContext.broadcast(coefficients)
    val dVectorBroadcast = input.sparkContext.broadcast(dVector)
    val aggregator = new HessianVectorAggregator(
      singleLossFunction,
      PhotonBroadcast(coefficientsBroadcast),
      PhotonBroadcast(dVectorBroadcast),
      normalizationContext)
    val resultAggregator = input.treeAggregate(aggregator)(
      seqOp = (ag, datum) => ag.add(datum),
      combOp = (ag1, ag2) => ag1.merge(ag2),
      depth = treeAggregateDepth
    )

    coefficientsBroadcast.unpersist()
    dVectorBroadcast.unpersist()

    resultAggregator.hessianD
  }

  /**
   * Calculate the multiplication between the Hessian matrix for an objective function and some vector d locally.
   *
   * @param input An iterable set of points
   * @param coefficients The current model coefficients
   * @param dVector The Hessian multiplication vector
   * @param singleLossFunction The function used to compute loss for predictions
   * @param normalizationContext The normalization context
   * @return The multiplication between the Hessian matrix and some vector d
   */
  def calcHessianVector(
      input: Iterable[LabeledPoint],
      coefficients: Vector[Double],
      dVector: Vector[Double],
      singleLossFunction: PointwiseLossFunction,
      normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] = {

    val aggregator = new HessianVectorAggregator(
      singleLossFunction,
      PhotonNonBroadcast(coefficients),
      PhotonNonBroadcast(dVector),
      normalizationContext)
    val resultAggregator = input.aggregate(aggregator)(
      seqop = (ag, datum) => ag.add(datum),
      combop = (ag1, ag2) => ag1.merge(ag2))

    resultAggregator.hessianD
  }
}
