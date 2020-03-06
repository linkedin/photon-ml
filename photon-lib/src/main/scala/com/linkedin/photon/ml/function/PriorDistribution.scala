/*
 * Copyright 2020 LinkedIn Corp. All rights reserved.
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

import breeze.linalg.{DenseMatrix, DenseVector, Vector, diag, sum}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.model.{Coefficients => ModelCoefficients}
import com.linkedin.photon.ml.util.{BroadcastWrapper, VectorUtils}

/**
 * Trait for an incremental training objective function. It is assumed that the prior is Gaussian Distribution with mean
 * as prior model's means and variance as prior model's variances. incrementalWeight controls importance of the prior
 * distribution. The larger incrementalWeight is, the more important the prior distribution is. incrementalWeight has a
 * default value of 1. l2RegWeight sets the L2 regularization term for any feature which is not included in the prior
 * model. For any feature which is included in the prior model, the equivalent L2 regularization term is
 * incrementalWeight / prior model's variance.
 */
trait PriorDistribution extends ObjectiveFunction {

  protected var l2RegWeight: Double = 0D
  protected var incrementalWeight: Double = 1D

  val priorCoefficients: ModelCoefficients = ModelCoefficients(DenseVector.zeros(1))

  lazy protected val priorMeans: Vector[Double] = priorCoefficients.means
  lazy protected val priorVariances: Vector[Double] = priorCoefficients.variancesOption.get
  lazy protected val inversePriorVariances: DenseVector[Double] = VectorUtils.invertVectorWithZeroHandler(priorVariances, l2RegWeight).toDenseVector

  require(l2RegWeight >= 0D, s"Invalid regularization weight '$l2RegWeight")

  /**
   * Compute the value of the function over the given data for the given model coefficients, with regularization towards
   * the prior coefficients.
   *
   * @param input The data over which to compute the objective function value
   * @param coefficients The model coefficients for which to compute the objective function's value
   * @param normalizationContext The normalization context
   * @return The value of the objective function and regularization terms
   */
  abstract override protected[ml] def value(
      input: Data,
      coefficients: Vector[Double],
      normalizationContext: BroadcastWrapper[NormalizationContext]): Double =
    super.value(input, coefficients, normalizationContext) + l2RegValue(coefficients)

  /**
   * Compute the Gaussian regularization term for the given model coefficients. L2 regularization term is
   * incrementalWeight * sum(pow(coefficients - priorMeans, 2) :/ priorVariance) / 2.
   *
   * @param coefficients The model coefficients
   * @return The Gaussian regularization term value
   */
  protected def l2RegValue(coefficients: Vector[Double]): Double = {

    val normalizedSquaredCoefficients = (coefficients - priorMeans) *:* inversePriorVariances *:* (coefficients - priorMeans)

    incrementalWeight * sum(normalizedSquaredCoefficients) / 2
  }
}

trait PriorDistributionDiff extends DiffFunction with PriorDistribution {

  /**
   * Compute the value of the function over the given data for the given model coefficients, with regularization towards
   * the prior coefficients.
   *
   * @param input The data over which to compute the objective function value
   * @param coefficients The model coefficients for which to compute the objective function's value
   * @param normalizationContext The normalization context
   * @return The value of the objective function and regularization terms
   */
  abstract override protected[ml] def value(
      input: Data,
      coefficients: Vector[Double],
      normalizationContext: BroadcastWrapper[NormalizationContext]): Double =
    calculate(input, coefficients, normalizationContext)._1

  /**
   * Compute the gradient of the function over the given data for the given model coefficients, with regularization
   * towards the prior coefficients.
   *
   * @param input The data over which to compute the objective function gradient
   * @param coefficients The model coefficients for which to compute the objective function's gradient
   * @param normalizationContext The normalization context
   * @return The gradient of the objective function and regularization terms
   */
  abstract override protected[ml] def gradient(
      input: Data,
      coefficients: Vector[Double],
      normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
    calculate(input, coefficients, normalizationContext)._2

  /**
   * Compute both the value and the gradient of the function over the given data for the given model coefficients, with
   * regularization towards the prior coefficients (computing value and gradient at once is more efficient than
   * computing them sequentially).
   *
   * @param input The data over which to compute the objective function value and gradient
   * @param coefficients The model coefficients for which to compute the objective function's value and gradient
   * @param normalizationContext The normalization context
   * @return The value and gradient of the objective function and regularization terms
   */
  abstract override protected[ml] def calculate(
      input: Data,
      coefficients: Vector[Double],
      normalizationContext: BroadcastWrapper[NormalizationContext]): (Double, Vector[Double]) = {

    val (baseValue, baseGradient) = super.calculate(input, coefficients, normalizationContext)
    val valueWithRegularization = baseValue + l2RegValue(coefficients)
    val gradientWithRegularization = baseGradient + l2RegGradient(coefficients)

    (valueWithRegularization, gradientWithRegularization)
  }

  /**
   * Compute the gradient of the Gaussian regularization term for the given model coefficients. Gradient is
   * incrementalWeight * (coefficients - priorMeans) :/ priorVariance.
   *
   * @param coefficients The model coefficients
   * @return The gradient of the Gaussian regularization term
   */
  protected def l2RegGradient(coefficients: Vector[Double]): Vector[Double] = {

    val normalizedCoefficients = (coefficients - priorMeans) *:* inversePriorVariances

    incrementalWeight * normalizedCoefficients
  }
}

trait PriorDistributionTwiceDiff extends TwiceDiffFunction with PriorDistributionDiff {

  /**
   * Compute the Hessian diagonal of the objective function over the given data for the given model coefficients, * the
   * gradient direction, with regularization towards the prior coefficients.
   *
   * @param input The data over which to compute the Hessian diagonal * gradient direction
   * @param coefficients The model coefficients for which to compute the objective function's Hessian diagonal
   *                     * gradient direction
   * @param multiplyVector The gradient direction vector
   * @param normalizationContext The normalization context
   * @return The Hessian diagonal (multiplied by the gradient direction) of the objective function and regularization
   *         terms
   */
  abstract override protected[ml] def hessianVector(
      input: Data,
      coefficients: Vector[Double],
      multiplyVector: Vector[Double],
      normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
    super.hessianVector(input, coefficients, multiplyVector, normalizationContext) +
      l2RegHessianVector(multiplyVector)

  /**
   * Compute the Hessian diagonal of the objective function over the given data for the given model coefficients, with
   * regularization towards the prior coefficients.
   *
   * @param input The data over which to compute the Hessian diagonal
   * @param coefficients The model coefficients for which to compute the objective function's Hessian diagonal
   * @return The Hessian diagonal of the objective function and regularization terms
   */
  abstract override protected[ml] def hessianDiagonal(input: Data, coefficients: Vector[Double]): Vector[Double] =
    super.hessianDiagonal(input, coefficients) :+ l2RegHessianDiagonal

  /**
   * Compute the Hessian matrix of the objective function over the given data for the given model coefficients, with
   * regularization towards the prior coefficients.
   *
   * @param input The data over which to compute the Hessian matrix
   * @param coefficients The model coefficients for which to compute the objective function's Hessian matrix
   * @return The Hessian matrix of the objective function and regularization terms
   */
  abstract override protected[ml] def hessianMatrix(input: Data, coefficients: Vector[Double]): DenseMatrix[Double] =
    super.hessianMatrix(input, coefficients) + l2RegHessianMatrix

  /**
   * Compute the Hessian diagonal * gradient direction of the Gaussian regularization term for the given model
   * coefficients.
   *
   * @param multiplyVector The gradient direction vector
   * @return The Hessian diagonal of the Gaussian regularization term, with gradient direction vector
   */
  protected def l2RegHessianVector(multiplyVector: Vector[Double]): Vector[Double] =
    incrementalWeight * (multiplyVector *:* inversePriorVariances)

  /**
   * Compute the Hessian diagonal of the Gaussian regularization term for the given model coefficients. Hessian
   * diagonal is incrementalWeight :/ priorVariance.
   *
   * @return The Hessian diagonal of the Gaussian regularization term
   */
  protected def l2RegHessianDiagonal: Vector[Double] = incrementalWeight * inversePriorVariances

  /**
   * Compute the Hessian matrix of the Gaussian regularization term for the given model coefficients.
   *
   * @return The Hessian matrix of the Gaussian regularization term
   */
  protected def l2RegHessianMatrix: DenseMatrix[Double] = incrementalWeight * diag(inversePriorVariances)
}
