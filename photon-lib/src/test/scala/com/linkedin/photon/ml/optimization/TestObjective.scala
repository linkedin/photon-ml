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
package com.linkedin.photon.ml.optimization

import breeze.linalg.{DenseMatrix, Vector, sum}

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.{ObjectiveFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.{BroadcastWrapper, VectorUtils}

/**
 * Test objective function used solely to exercise the optimizers.
 */
class TestObjective extends ObjectiveFunction with TwiceDiffFunction {

  type Data = Iterable[LabeledPoint]
  type Coefficients = Vector[Double]

  // These 3 methods are copied directly from the SingleNodeObjectiveFunction from photon-api.
  override protected[ml] def domainDimension(input: Iterable[LabeledPoint]): Int = input.head.features.size
  override protected[ml] def convertFromVector(coefficients: Vector[Double]): Coefficients = coefficients
  override protected[ml] def convertToVector(coefficients: Vector[Double]): Vector[Double] = coefficients

  /**
   * Compute the value of the function over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the objective value
   * @param coefficients The model coefficients used to compute the function's value
   * @param normalizationContext The normalization context
   * @return The computed value of the function
   */
  override protected[ml] def value(
    input: Iterable[LabeledPoint],
    coefficients: Vector[Double],
    normalizationContext: BroadcastWrapper[NormalizationContext]): Double =
    calculate(input, coefficients, normalizationContext)._1

  /**
   * Compute the gradient of the function over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the gradient
   * @param coefficients The model coefficients used to compute the function's gradient
   * @param normalizationContext The normalization context
   * @return The computed gradient of the function
   */
  override protected[ml] def gradient(input: Iterable[LabeledPoint], coefficients: Vector[Double],
    normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
    calculate(input, coefficients, normalizationContext)._2

  /**
   * Compute both the value and the gradient of the function for the given model coefficients (computing value and
   * gradient at once is sometimes more efficient than computing them sequentially).
   *
   * @param input The given data over which to compute the value and gradient
   * @param coefficients The model coefficients used to compute the function's value and gradient
   * @param normalizationContext The normalization context
   * @return The computed value and gradient of the function
   */
  override protected[ml] def calculate(
    input: Iterable[LabeledPoint],
    coefficients: Vector[Double],
    normalizationContext: BroadcastWrapper[NormalizationContext]): (Double, Vector[Double]) = {

    val initialCumGradient = VectorUtils.zeroOfSameType(coefficients)

    input.aggregate((0.0, initialCumGradient))(
      seqop = {
        case ((loss, cumGradient), datum) =>
          val v = TestObjective.calculateAt(datum, coefficients, cumGradient)
          (loss + v, cumGradient)
      },
      combop = {
        case ((loss1, grad1), (loss2, grad2)) =>
          (loss1 + loss2, grad1 += grad2)
      })
  }

  /**
   * Compute (Hessian * d_i) over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the Hessian
   * @param coefficients The model coefficients used to compute the function's hessian, multiplied by a given vector
   * @param multiplyVector The given vector to be dot-multiplied with the Hessian. For example, in conjugate
   *                       gradient method this would correspond to the gradient multiplyVector.
   * @param normalizationContext The normalization context
   * @return The computed Hessian multiplied by the given multiplyVector
   */
  override protected[ml] def hessianVector(
    input: Iterable[LabeledPoint],
    coefficients: Vector[Double],
    multiplyVector: Vector[Double],
    normalizationContext: BroadcastWrapper[NormalizationContext]) : Vector[Double] = {

    val initialCumHessianVector = VectorUtils.zeroOfSameType(coefficients)

    input.aggregate(initialCumHessianVector)(
      seqop = (cumHessianVector, datum) => {
        TestObjective.hessianVectorAt(datum, coefficients, multiplyVector, cumHessianVector)
        cumHessianVector
      },
      combop = _ += _)
  }

  /**
   * Unused, only implemented as part of TwiceDiffFunction.
   */
  override protected[ml] def hessianDiagonal(
      input: Iterable[LabeledPoint],
      coefficients: Vector[Double]): Vector[Double] =
    Coefficients.initializeZeroCoefficients(coefficients.length).means

  /**
   * Unused, only implemented as part of TwiceDiffFunction.
   */
  override protected[ml] def hessianMatrix(
      input: Iterable[LabeledPoint],
      coefficients: Vector[Double]): DenseMatrix[Double] =
    DenseMatrix.zeros[Double](coefficients.length, coefficients.length)

}

object TestObjective {

  val CENTROID = 4.0

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
    val deltaSq = delta.mapValues { x => x * x }
    cumGradient += delta :* 2.0
    sum(deltaSq)
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
    cumHessianVector: Vector[Double]): Unit = cumHessianVector += vector :* 2.0
}
