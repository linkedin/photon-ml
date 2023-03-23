/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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

import breeze.linalg._
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.glm.PointwiseLossFunction
import com.linkedin.photon.ml.util.{BroadcastWrapper, PhotonBroadcast, PhotonNonBroadcast}

/**
 * An aggregator to calculate the diagonal of the Hessian matrix of a generalized linear model loss function on a
 *
 * This class is heavily influenced by [[ValueAndGradientAggregator]] and [[HessianVectorAggregator]]. Note that unlike
 * those two classes, this class does not handle normalization.
 *
 * @param lossFunction A single loss function for the generalized linear model
 * @param coefficients The coefficients for the GLM
 */
@SerialVersionUID(1L)
protected[ml] class HessianDiagonalAggregator(
    lossFunction: PointwiseLossFunction,
    coefficients: BroadcastWrapper[Vector[Double]]) extends Serializable {

  /**
   * The equation for computing the Hessian matrix for a particular set of coefficients over a dataset is as follows:
   *
   * H_jk = ∂^2^ L(w, X, y, u, o) / (∂ w_j)(∂ w_k)
   *      = \sum_i ∂^2^ (u_i * l(z_i, y_i)) / (∂ w_j)(∂ w_k)
   *      = \sum_i (∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i * X_i_j * X_i_k
   *
   * Therefore, for the diagonal h of the Hessian matrix (j = k):
   *
   * h = \sum_i e((∂^2^ l(z_i, y_i) / (∂ z_i)^2^) * u_i, m) *:* X_i *:* X_i
   *
   * where:
   *
   * e(a, b) = A vector of length b where each value is a
   * m = The dimension of all X (and therefore w as well as H, which is square)
   *
   * vectorSum is the aggregator value for vector h.
   */
  protected var vectorSum: Vector[Double] = DenseVector.zeros[Double](coefficients.value.length)

  /**
   * Getter for the cumulative Hessian diagonal.
   *
   * @return The cumulative Hessian diagonal
   */
  def hessianDiagonal: Vector[Double] = vectorSum

  /**
   * Add a data point to the aggregator.
   *
   * @param datum The data point
   * @return This aggregator object itself
   */
  def add(datum: LabeledPoint): this.type = {

    val LabeledPoint(label, features, _, weight) = datum

    require(
      features.length == coefficients.value.length,
      s"Size mismatch: Coefficient size = ${coefficients.value.length}, features size = ${features.size}")

    val margin = datum.computeMargin(coefficients.value)
    val dzzLoss = lossFunction.DzzLoss(margin, label)

    axpy(weight * dzzLoss, features *:* features, vectorSum)

    this
  }

  /**
   * Merge two aggregators.
   *
   * @param that The other aggregator
   * @return This aggregator object itself, with the contents of the other aggregator object merged into it
   */
  def merge(that: HessianDiagonalAggregator): this.type = {

    // TODO: The below tests are technically correct, but unnecessarily slow. Currently the only time that two
    // TODO: aggregators are merged is during aggregation, where they are copies of the same initial object.
//    require(lossFunction.equals(that.lossFunction), "Attempting to merge aggregators with different loss functions")
//    require(coefficients == that.coefficients, "Attempting to merge aggregators with different coefficients vectors")

    vectorSum :+= that.vectorSum

    this
  }
}

object HessianDiagonalAggregator {

  /**
   * Calculate the Hessian diagonal for an objective function in Spark.
   *
   * @param input An RDD of data points
   * @param coefficients The current model coefficients
   * @param singleLossFunction The GLM loss function
   * @param treeAggregateDepth The tree aggregate depth
   * @return The Hessian vector
   */
  def calcHessianDiagonal(
      input: RDD[LabeledPoint],
      coefficients: Vector[Double],
      singleLossFunction: PointwiseLossFunction,
      treeAggregateDepth: Int): Vector[Double] = {

    val coefficientsBroadcast = input.sparkContext.broadcast(coefficients)
    val aggregator = new HessianDiagonalAggregator(singleLossFunction, PhotonBroadcast(coefficientsBroadcast))
    val resultAggregator = input.treeAggregate(aggregator)(
      seqOp = (ag, datum) => ag.add(datum),
      combOp = (ag1, ag2) => ag1.merge(ag2),
      depth = treeAggregateDepth)

    coefficientsBroadcast.unpersist()

    resultAggregator.hessianDiagonal
  }

  /**
   * Calculate the Hessian diagonal for an objective function locally.
   *
   * @param input An iterable set of points
   * @param coefficients The current model coefficients
   * @param singleLossFunction The GLM loss function
   * @return The Hessian vector
   */
  def calcHessianDiagonal(
      input: Iterable[LabeledPoint],
      coefficients: Vector[Double],
      singleLossFunction: PointwiseLossFunction): Vector[Double] = {

    val aggregator = new HessianDiagonalAggregator(singleLossFunction, PhotonNonBroadcast(coefficients))
    val resultAggregator = input.aggregate(aggregator)(
      seqop = (ag, datum) => ag.add(datum),
      combop = (ag1, ag2) => ag1.merge(ag2))

    resultAggregator.hessianDiagonal
  }
}

