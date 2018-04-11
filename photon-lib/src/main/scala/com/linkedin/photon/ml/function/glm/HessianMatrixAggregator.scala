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

import breeze.linalg._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint

/**
 * An aggregator to perform calculation of the Hessian matrix. Both Iterable and RDD data share the same logic for data
 * aggregation.
 *
 * @param func A single loss function for the generalized linear model
 * @param dim The dimension (number of features) of the aggregator
 */
@SerialVersionUID(1L)
protected[ml] class HessianMatrixAggregator(func: PointwiseLossFunction, val dim: Int) extends Serializable {

  protected var matrixSum: DenseMatrix[Double] = DenseMatrix.zeros[Double](dim, dim)

  /**
   * Return the cumulative Hessian matrix.
   *
   * @return The cumulative Hessian diagonal
   */
  def getMatrix: DenseMatrix[Double] = matrixSum

  /**
   * Add a data point to the aggregator.
   *
   * @param datum The data point
   * @param coefficients The current model coefficients
   * @return The aggregator itself
   */
  def add(datum: LabeledPoint, coefficients: Vector[Double]): this.type = {
    val LabeledPoint(label, features, _, weight) = datum

    require(features.size == dim, s"Size mismatch. Coefficient size: $dim, features size: ${features.size}")

    val margin = datum.computeMargin(coefficients)
    val dzzLoss = func.DzzLoss(margin, label)

    // Convert features to a dense matrix so that we can compute the outer product
    val x = features.toDenseVector.asDenseMatrix
    val hessianMatrix = dzzLoss * (x.t * x)

    axpy(weight, hessianMatrix, matrixSum)

    this
  }

  /**
   * Merge two aggregators.
   *
   * @param that The other aggregator
   * @return A merged aggregator
   */
  def merge(that: HessianMatrixAggregator): this.type = {
    require(dim == that.dim, s"Dimension mismatch. this.dim=$dim, that.dim=${that.dim}")
    require(that.getClass.eq(getClass), s"Class mismatch. this.class=$getClass, that.class=${that.getClass}")

    axpy(1.0, that.matrixSum, matrixSum)

    this
  }
}

object HessianMatrixAggregator {
  /**
   * Calculate the Hessian matrix for an objective function in Spark.
   *
   * @param input An RDD of data points
   * @param coef The current model coefficients
   * @param singleLossFunction The function used to compute loss for predictions
   * @param treeAggregateDepth The tree aggregate depth
   * @return The Hessian matrix
   */
  def calcHessianMatrix(
    input: RDD[LabeledPoint],
    coef: Broadcast[Vector[Double]],
    singleLossFunction: PointwiseLossFunction,
    treeAggregateDepth: Int): DenseMatrix[Double] = {

    val aggregator = new HessianMatrixAggregator(singleLossFunction, coef.value.size)
    val resultAggregator = input.treeAggregate(aggregator)(
      seqOp = (ag, datum) => ag.add(datum, coef.value),
      combOp = (ag1, ag2) => ag1.merge(ag2),
      depth = treeAggregateDepth
    )

    resultAggregator.getMatrix
  }

  /**
   * Calculate the Hessian matrix for an objective function locally.
   *
   * @param input An iterable set of points
   * @param coef The current model coefficients
   * @param singleLossFunction The function used to compute loss for predictions
   * @return The Hessian matrix
   */
  def calcHessianMatrix(
    input: Iterable[LabeledPoint],
    coef: Vector[Double],
    singleLossFunction: PointwiseLossFunction): DenseMatrix[Double] = {

    val aggregator = new HessianMatrixAggregator(singleLossFunction, coef.size)
    val resultAggregator = input.aggregate(aggregator)(
      seqop = (ag, datum) => ag.add(datum, coef),
      combop = (ag1, ag2) => ag1.merge(ag2)
    )

    resultAggregator.getMatrix
  }
}
