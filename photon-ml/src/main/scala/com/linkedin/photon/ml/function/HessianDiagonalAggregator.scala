/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.function

import breeze.linalg._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.glm.PointwiseLossFunction

/**
 * An aggregator to perform calculation of the Hessian diagonal vector. Both Iterable and RDD data share the same logic
 * for data aggregation.
 *
 * This class is heavily influenced by the HessianVectorAggregator and the ValueAndGradientAggregator.
 *
 * @param func A single loss function for the generalized linear model
 * @param dim The dimension (number of features) of the aggregator
 */
@SerialVersionUID(1L)
protected[ml] class HessianDiagonalAggregator(func: PointwiseLossFunction, val dim: Int) extends Serializable {

  protected var vectorSum: Vector[Double] = DenseVector.zeros[Double](dim)

  /**
   * Return the cumulative Hessian diagonal.
   *
   * @return The cumulative Hessian diagonal
   */
  def getVector: Vector[Double] = vectorSum

  /**
   * Add a data point to the aggregator
   *
   * @param datum The data point
   * @param coefficients The current model coefficients
   * @return The aggregator itself
   */
  def add(datum: LabeledPoint, coefficients: Vector[Double]): this.type = {
    val LabeledPoint(label, features, _, weight) = datum

    require(features.size == dim, s"Size mismatch. Coefficient size: $dim, features size: ${features.size}")

    val margin = datum.computeMargin(coefficients)
    val d2ldz2 = func.d2lossdz2(margin, label)

    axpy(weight * d2ldz2, features :* features, vectorSum)
    this
  }

  /**
   * Merge two aggregators
   *
   * @param that The other aggregator
   * @return A merged aggregator
   */
  def merge(that: HessianDiagonalAggregator): this.type = {
    require(dim == that.dim, s"Dimension mismatch. this.dim=$dim, that.dim=${that.dim}")
    require(that.getClass.eq(getClass), s"Class mismatch. this.class=$getClass, that.class=${that.getClass}")

    axpy(1.0, that.vectorSum, vectorSum)

    this
  }
}

object HessianDiagonalAggregator {
  /**
   * Calculate the Hessian diagonal for an objective function in Spark
   *
   * @param input An RDD of data points
   * @param coef The current model coefficients
   * @param singleLossFunction The function used to compute loss for predictions
   * @param treeAggregateDepth The tree aggregate depth
   * @return The Hessian vector
   */
  def calcHessianDiagonal(
    input: RDD[LabeledPoint],
    coef: Broadcast[Vector[Double]],
    singleLossFunction: PointwiseLossFunction,
    treeAggregateDepth: Int): Vector[Double] = {

    val aggregator = new HessianDiagonalAggregator(singleLossFunction, coef.value.size)
    val resultAggregator = input.treeAggregate(aggregator)(
      seqOp = (ag, datum) => ag.add(datum, coef.value),
      combOp = (ag1, ag2) => ag1.merge(ag2),
      depth = treeAggregateDepth
    )

    resultAggregator.getVector
  }

  /**
   * Calculate the Hessian diagonal for an objective function locally
   *
   * @param input An iterable set of points
   * @param coef The current model coefficients
   * @param singleLossFunction The function used to compute loss for predictions
   * @return The Hessian vector
   */
  def calcHessianDiagonal(
    input: Iterable[LabeledPoint],
    coef: Vector[Double],
    singleLossFunction: PointwiseLossFunction): Vector[Double] = {

    val aggregator = new HessianDiagonalAggregator(singleLossFunction, coef.size)
    val resultAggregator = input.aggregate(aggregator)(
      seqop = (ag, datum) => ag.add(datum, coef),
      combop = (ag1, ag2) => ag1.merge(ag2)
    )

    resultAggregator.getVector
  }
}

