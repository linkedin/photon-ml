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
package com.linkedin.photon.ml.optimization

import breeze.linalg.{Vector, sum}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.{DistributedObjectiveFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.Utils

/**
 * Test function used solely to exercise the optimizers.
 *
 * This function has known minimum at {@link IntegTestObjective.CENTROID}.
 */
class IntegTestObjective(sc: SparkContext, treeAggregateDepth: Int)
  extends DistributedObjectiveFunction(sc, treeAggregateDepth)
  with TwiceDiffFunction {

  override protected[ml] def value(
    input: RDD[LabeledPoint],
    coefficients: Broadcast[Vector[Double]],
    normalizationContext: Broadcast[NormalizationContext]): Double =
    calculate(input, coefficients, normalizationContext)._1

  override protected[ml] def gradient(
    input: RDD[LabeledPoint],
    coefficients: Broadcast[Vector[Double]],
    normalizationContext: Broadcast[NormalizationContext]): Vector[Double] =
    calculate(input, coefficients, normalizationContext)._2

  override protected[ml] def calculate(
    input: RDD[LabeledPoint],
    coefficients: Broadcast[Vector[Double]],
    normalizationContext: Broadcast[NormalizationContext]): (Double, Vector[Double]) = {

    val initialCumGradient = Utils.initializeZerosVectorOfSameType(coefficients.value)

    input.treeAggregate((0.0, initialCumGradient))(
      seqOp = {
        case ((loss, cumGradient), datum) =>
          val v = IntegTestObjective.calculateAt(datum, coefficients.value, cumGradient)
          (loss + v, cumGradient)
      },
      combOp = {
        case ((loss1, grad1), (loss2, grad2)) =>
          (loss1 + loss2, grad1 += grad2)
      })
  }

  override protected[ml] def hessianVector(
    input: RDD[LabeledPoint],
    coefficients: Broadcast[Vector[Double]],
    multiplyVector: Broadcast[Vector[Double]],
    normalizationContext: Broadcast[NormalizationContext]) : Vector[Double] = {

    val initialCumHessianVector = Utils.initializeZerosVectorOfSameType(coefficients.value)

    input.treeAggregate(initialCumHessianVector)(
      seqOp = (cumHessianVector, datum) => {
        IntegTestObjective.hessianVectorAt(datum, coefficients.value, multiplyVector.value, cumHessianVector)
        cumHessianVector
      },
      combOp = _ += _,
      treeAggregateDepth)
  }

  /**
   * Unused, only implemented as part of TwiceDiffFunction.
   */
  override protected[ml] def hessianDiagonal(
    input: RDD[LabeledPoint],
    coefficients: Broadcast[Vector[Double]]): Vector[Double] =
    Coefficients.initializeZeroCoefficients(coefficients.value.size).means
}

object IntegTestObjective {
  val CENTROID = Math.PI

  /**
   * Compute the value and gradient at a single data point. Since the function has known minimum, the input data is
   * ignored.
   *
   * @param dataPoint A single data point
   * @param coefficients The current coefficients
   * @param cumGradient The cumulative Gradient vector for all points in the dataset
   * @return The value at the given data point
   */
  protected def calculateAt(
    dataPoint: LabeledPoint,
    coefficients: Vector[Double],
    cumGradient: Vector[Double]): Double = {

    val delta = coefficients - CENTROID
    val expDeltaSq = delta.mapValues { x => Math.exp(Math.pow(x, 2.0)) }
    cumGradient += expDeltaSq :* delta :* 2.0
    sum(expDeltaSq) - expDeltaSq.length
  }

  /**
   * Compute the Hessian vector at a single data point. Since the function has known minimum, the input data is ignored.
   *
   * @param dataPoint A single data point
   * @param coefficients The current coefficients
   * @param vector The Hessian multiply vector
   * @param cumHessianVector The cumulative Hessian vector for all points in the dataset
   */
  protected def hessianVectorAt(
    dataPoint: LabeledPoint,
    coefficients: Vector[Double],
    vector: Vector[Double],
    cumHessianVector: Vector[Double]): Unit = {

    val delta = coefficients - CENTROID
    val expDeltaSq = delta.mapValues { x => Math.exp(Math.pow(x, 2.0)) }
    val hess = expDeltaSq :* (delta :* delta :+ 1.0) :* 4.0
    cumHessianVector += hess :* vector
  }
}
