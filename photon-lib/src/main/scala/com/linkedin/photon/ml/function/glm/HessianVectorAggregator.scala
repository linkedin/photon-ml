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
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper

// TODO: Better document this algorithm, especially normalization.
/**
 * An aggregator to perform calculation of Hessian vector multiplication for generalized linear model loss functions,
 * especially in the context of normalization. Both Iterable and RDD data share the same logic for data aggregation.
 *
 * Some logic of Hessian vector multiplication is the same for gradient aggregation, so this class inherits
 * ValueAndGradientAggregator.
 *
 * @param func A single loss function for the generalized linear model
 * @param dim The dimension (number of features) of the aggregator
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

  /**
   * Initialize the aggregator with proper multiply vector and product shifts if normalization is used.
   *
   * @param coef The current model coefficients
   * @param multiplyVector The Hessian multiplication vector
   * @param normalizationContext The normalization context
   */
  def init(coef: Vector[Double], multiplyVector: Vector[Double], normalizationContext: NormalizationContext): Unit = {
    super.init(coef, normalizationContext)

    require(multiplyVector.size == dim, s"Size mismatch. Multiply vector size: ${multiplyVector.size} != $dim.")

    val NormalizationContext(factorsOption, shiftsOption, interceptIdOption) = normalizationContext
    effectiveMultiplyVector = factorsOption match {
      case Some(factors) =>
        interceptIdOption.foreach(id =>
          require(
            factors(id) == 1.0,
            s"The intercept should not be transformed. Intercept scaling factor: ${factors(id)}"))
        require(factors.size == dim, s"Size mismatch. Factors vector size: ${factors.size} != $dim.")

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
   * Add a data point to the aggregator.
   *
   * @param datum The data point
   * @param coef The current model coefficients
   * @param multiplyVector The Hessian multiplication vector
   * @param normalizationContext The normalization context
   * @return The aggregator itself
   */
  def add(
    datum: LabeledPoint,
    coef: Vector[Double],
    multiplyVector: Vector[Double],
    normalizationContext: NormalizationContext): this.type = {

    if (!initialized) {
      this.synchronized {
        init(coef, multiplyVector, normalizationContext)
        initialized = true
      }
    }

    val LabeledPoint(label, features, _, weight) = datum
    require(features.size == dim, s"Size mismatch. Coefficient size: $dim, features size: ${features.size}")
    val margin = datum.computeMargin(effectiveCoefficients) + marginShift
    val dzzLoss = func.DzzLoss(margin, label)
    // l'' * (\sum_k x_{ki} * effectiveMultiplyVector_k - featureVectorProductShift)
    val effectiveWeight = weight * dzzLoss * (features.dot(effectiveMultiplyVector) - featureVectorProductShift)

    totalCnt += 1
    vectorShiftPrefactorSum += effectiveWeight
    axpy(effectiveWeight, features, vectorSum)

    this
  }
}

object HessianVectorAggregator {
  /**
   * Calculate the Hessian vector for an objective function in Spark.
   *
   * @param input An RDD of data points
   * @param coef The current model coefficients
   * @param multiplyVector The Hessian multiplication vector
   * @param singleLossFunction The function used to compute loss for predictions
   * @param normalizationContext The normalization context
   * @param treeAggregateDepth The tree aggregate depth
   * @return The Hessian vector
   */
  def calcHessianVector(
      input: RDD[LabeledPoint],
      coef: Broadcast[Vector[Double]],
      multiplyVector: Broadcast[Vector[Double]],
      singleLossFunction: PointwiseLossFunction,
      normalizationContext: BroadcastWrapper[NormalizationContext],
      treeAggregateDepth: Int): Vector[Double] = {

    val aggregator = new HessianVectorAggregator(singleLossFunction, coef.value.size)
    val resultAggregator = input.treeAggregate(aggregator)(
      seqOp = (ag, datum) => ag.add(datum, coef.value, multiplyVector.value, normalizationContext.value),
      combOp = (ag1, ag2) => ag1.merge(ag2),
      depth = treeAggregateDepth
    )

    resultAggregator.getVector(normalizationContext.value)
  }

  /**
   * Calculate the Hessian vector for an objective function locally.
   *
   * @param input An iterable set of points
   * @param coef The current model coefficients
   * @param multiplyVector The Hessian multiplication vector
   * @param singleLossFunction The function used to compute loss for predictions
   * @param normalizationContext The normalization context
   * @return The Hessian vector
   */
  def calcHessianVector(
      input: Iterable[LabeledPoint],
      coef: Vector[Double],
      multiplyVector: Vector[Double],
      singleLossFunction: PointwiseLossFunction,
      normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] = {

    val aggregator = new HessianVectorAggregator(singleLossFunction, coef.size)
    val resultAggregator = input.aggregate(aggregator)(
      seqop = (ag, datum) => ag.add(datum, coef, multiplyVector, normalizationContext.value),
      combop = (ag1, ag2) => ag1.merge(ag2)
    )

    resultAggregator.getVector(normalizationContext.value)
  }
}
