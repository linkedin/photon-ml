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
import com.linkedin.photon.ml.function.{ObjectiveFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.VectorUtils

/**
 * Test function used solely to exercise the optimizers.
 *
 * This function has known minimum at {@link IntegTestObjective.CENTROID}.
 */
class IntegTestObjective(sc: SparkContext, treeAggregateDepth: Int) extends ObjectiveFunction with TwiceDiffFunction {

  type Data = RDD[LabeledPoint]
  type Coefficients = Broadcast[Vector[Double]]

  // These 4 methods are copied directly from [[DistributedObjectiveFunction]] in photon-api.
  override protected[ml] def domainDimension(input: Data): Int = input.first.features.size
  override protected[ml] def convertFromVector(coefficients: Vector[Double]): Coefficients = sc.broadcast(coefficients)
  override protected[ml] def convertToVector(coefficients: Coefficients): Vector[Double] = coefficients.value
  override protected[ml] def cleanupCoefficients(coefficients: Coefficients): Unit = coefficients.unpersist()

  /**
   *
   * @param input The given data over which to compute the objective value
   * @param coefficients The model coefficients used to compute the function's value
   * @param normalizationContext The normalization context
   * @return The computed value of the function
   */
  override protected[ml] def value(
      input: RDD[LabeledPoint],
      coefficients: Broadcast[Vector[Double]],
      normalizationContext: Broadcast[NormalizationContext]): Double =
    calculate(input, coefficients, normalizationContext)._1

  /**
   *
   * @param input The given data over which to compute the gradient
   * @param coefficients The model coefficients used to compute the function's gradient
   * @param normalizationContext The normalization context
   * @return The computed gradient of the function
   */
  override protected[ml] def gradient(
      input: RDD[LabeledPoint],
      coefficients: Broadcast[Vector[Double]],
      normalizationContext: Broadcast[NormalizationContext]): Vector[Double] =
    calculate(input, coefficients, normalizationContext)._2

  /**
   *
   * @param input The given data over which to compute the value and gradient
   * @param coefficients The model coefficients used to compute the function's value and gradient
   * @param normalizationContext The normalization context
   * @return The computed value and gradient of the function
   */
  override protected[ml] def calculate(
      input: RDD[LabeledPoint],
      coefficients: Broadcast[Vector[Double]],
      normalizationContext: Broadcast[NormalizationContext]): (Double, Vector[Double]) = {

    val initialCumGradient = VectorUtils.initializeZerosVectorOfSameType(coefficients.value)

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

  /**
   *
   * @param input The given data over which to compute the Hessian
   * @param coefficients The model coefficients used to compute the function's hessian, multiplied by a given vector
   * @param multiplyVector The given vector to be dot-multiplied with the Hessian. For example, in conjugate
   *                       gradient method this would correspond to the gradient multiplyVector.
   * @param normalizationContext The normalization context
   * @return The computed Hessian multiplied by the given multiplyVector
   */
  override protected[ml] def hessianVector(
      input: RDD[LabeledPoint],
      coefficients: Broadcast[Vector[Double]],
      multiplyVector: Broadcast[Vector[Double]],
      normalizationContext: Broadcast[NormalizationContext]) : Vector[Double] = {

    val initialCumHessianVector = VectorUtils.initializeZerosVectorOfSameType(coefficients.value)

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
   *
   * @param input The given data over which to compute the diagonal of the Hessian matrix
   * @param coefficients The model coefficients used to compute the diagonal of the Hessian matrix
   * @return The computed diagonal of the Hessian matrix
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
