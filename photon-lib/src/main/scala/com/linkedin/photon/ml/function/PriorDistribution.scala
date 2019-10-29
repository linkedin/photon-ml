/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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
import breeze.numerics.{abs, sqrt}

import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.model.{Coefficients => ModelCoefficients}
import com.linkedin.photon.ml.util.{BroadcastWrapper, VectorUtils}

/**
 * Trait for an incremental training objective function. It is assumed that the prior is a product of Gaussian and
 * Laplace distributions. The L1 regularization weight refers to the relative weight of the Laplace prior. The L2
 * regularization weight refers to the relative weight of the Gaussian prior.
 */
trait PriorDistribution extends ObjectiveFunction {

  val priorCoefficients: ModelCoefficients = ModelCoefficients(DenseVector.zeros(1))

  lazy protected val priorMeans: Vector[Double] = priorCoefficients.means
  lazy protected val priorVariances: Vector[Double] = priorCoefficients.variancesOption.get
  lazy protected val inversePriorVariances: DenseVector[Double] = VectorUtils.invertVector(priorVariances).toDenseVector
  protected var l1RegWeight: Double = 0D
  protected var l2RegWeight: Double = 0D

  require(l1RegWeight >= 0D, s"Invalid regularization weight '$l1RegWeight")
  require(l2RegWeight >= 0D, s"Invalid regularization weight '$l2RegWeight")

  /**
   * Getter for the Laplace weight of the prior.
   *
   * @return The L1 regularization weight
   */
  def l1RegularizationWeight: Double = l1RegWeight

  /**
   * Getter for the Gaussian weight of the prior.
   *
   * @return The L2 regularization weight
   */
  def l2RegularizationWeight: Double = l2RegWeight

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
      coefficients: Coefficients,
      normalizationContext: BroadcastWrapper[NormalizationContext]): Double =
    super.value(input, coefficients, normalizationContext) +
      l1RegValue(convertToVector(coefficients)) +
      l2RegValue(convertToVector(coefficients))

  /**
   * Compute the Laplace regularization term for the given model coefficients. L1 regularization term is
   * l1RegWeight * sum(abs(coefficients - priorMeans) :/ sqrt(priorVariance)).
   *
   * @param coefficients The model coefficients
   * @return The Laplace regularization term value
   */
  protected def l1RegValue(coefficients: Vector[Double]): Double = {

    val normalizedCoefficients = (coefficients - priorMeans) :/ sqrt(priorVariances)

    l1RegWeight * sum(abs(normalizedCoefficients))
  }

  /**
   * Compute the Gaussian regularization term for the given model coefficients. L2 regularization term is
   * l2RegWeight * sum(pow(coefficients - priorMeans, 2) :/ priorVariance) / 2.
   *
   * @param coefficients The model coefficients
   * @return The Gaussian regularization term value
   */
  protected def l2RegValue(coefficients: Vector[Double]): Double = {

    val normalizedCoefficients = (coefficients - priorMeans) :/ sqrt(priorVariances)

    l2RegWeight * normalizedCoefficients.dot(normalizedCoefficients) / 2
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
      coefficients: Coefficients,
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
      coefficients: Coefficients,
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
      coefficients: Coefficients,
      normalizationContext: BroadcastWrapper[NormalizationContext]): (Double, Vector[Double]) = {

    val (baseValue, baseGradient) = super.calculate(input, coefficients, normalizationContext)
    val valueWithRegularization = baseValue + l1RegValue(convertToVector(coefficients)) +
      l2RegValue(convertToVector(coefficients))
    val gradientWithRegularization = baseGradient + l1RegGradient(convertToVector(coefficients)) +
      l2RegGradient(convertToVector(coefficients))

    (valueWithRegularization, gradientWithRegularization)
  }

  /**
   * Compute the gradient of the Laplace term for the given model coefficients. Gradient is
   * l1RegWeight :/ sqrt(priorVariance) if coefficients >= priorMeans;
   * - l1RegWeight :/ sqrt(priorVariance) if coefficients < priorMeans.
   *
   * @param coefficients The model coefficients
   * @return The gradient of the Laplace regularization term
   */
  protected def l1RegGradient(coefficients: Vector[Double]): Vector[Double] = {

    val coefficientsMask = (coefficients - priorMeans).map(coefficient => if (coefficient > 0) 1.0 else -1.0)

    l1RegWeight * (coefficientsMask :/ sqrt(priorVariances))
  }

  /**
   * Compute the gradient of the Gaussian regularization term for the given model coefficients. Gradient is
   * l2RegWeight * (coefficients - priorMeans) :/ priorVariance.
   *
   * @param coefficients The model coefficients
   * @return The gradient of the Gaussian regularization term
   */
  protected def l2RegGradient(coefficients: Vector[Double]): Vector[Double] = {

    val normalizedCoefficients = (coefficients - priorMeans) :/ priorVariances

    l2RegWeight * normalizedCoefficients
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
      coefficients: Coefficients,
      multiplyVector: Coefficients,
      normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
    super.hessianVector(input, coefficients, multiplyVector, normalizationContext) +
      l2RegHessianVector(convertToVector(multiplyVector))

  /**
   * Compute the Hessian diagonal of the objective function over the given data for the given model coefficients, with
   * regularization towards the prior coefficients.
   *
   * @param input The data over which to compute the Hessian diagonal
   * @param coefficients The model coefficients for which to compute the objective function's Hessian diagonal
   * @return The Hessian diagonal of the objective function and regularization terms
   */
  abstract override protected[ml] def hessianDiagonal(input: Data, coefficients: Coefficients): Vector[Double] =
    super.hessianDiagonal(input, coefficients) :+ l2RegHessianDiagonal

  /**
   * Compute the Hessian matrix of the objective function over the given data for the given model coefficients, with
   * regularization towards the prior coefficients.
   *
   * @param input The data over which to compute the Hessian matrix
   * @param coefficients The model coefficients for which to compute the objective function's Hessian matrix
   * @return The Hessian matrix of the objective function and regularization terms
   */
  abstract override protected[ml] def hessianMatrix(input: Data, coefficients: Coefficients): DenseMatrix[Double] =
    super.hessianMatrix(input, coefficients) + l2RegHessianMatrix

  /**
   * Compute the Hessian diagonal * gradient direction of the Gaussian regularization term for the given model
   * coefficients.
   *
   * @param multiplyVector The gradient direction vector
   * @return The Hessian diagonal of the Gaussian regularization term, with gradient direction vector
   */
  protected def l2RegHessianVector(multiplyVector: Vector[Double]): Vector[Double] =
    l2RegWeight * (multiplyVector /:/ priorVariances)

  /**
   * Compute the Hessian diagonal of the Gaussian regularization term for the given model coefficients. Hessian
   * diagonal is l2RegWeight :/ priorVariance.
   *
   * @return The Hessian diagonal of the Gaussian regularization term
   */
  protected def l2RegHessianDiagonal: Vector[Double] = l2RegWeight * inversePriorVariances

  /**
   * Compute the Hessian matrix of the Gaussian regularization term for the given model coefficients.
   *
   * @return The Hessian matrix of the Gaussian regularization term
   */
  protected def l2RegHessianMatrix: DenseMatrix[Double] = l2RegWeight * diag(inversePriorVariances)
}
