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

import breeze.linalg.Vector
import org.apache.spark.broadcast.Broadcast

import com.linkedin.photon.ml.normalization.NormalizationContext

/**
 * Trait for an objective function with L2 regularization.
 */
trait L2Regularization extends ObjectiveFunction {
  protected var l2RegWeight: Double = 0D

  /**
   * Getter.
   *
   * @return The L2 regularization weight
   */
  def l2RegularizationWeight: Double = l2RegWeight

  /**
   * Setter.
   *
   * Note: This function definition uses the setter syntactic sugar trick. Statements like:
   *
   *    objectiveFunction.l2RegularizationWeight = 10
   *
   * will call this function.
   *
   * @param newRegWeight The new L2 regularization weight
   */
  protected[ml] def l2RegularizationWeight_=(newRegWeight: Double): Unit = l2RegWeight = newRegWeight

  /**
   * Compute the value of the function with L2 regularization over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the objective value
   * @param coefficients The model coefficients used to compute the function's value
   * @param normalizationContext The normalization context
   * @return The computed value of the function
   */
  abstract override protected[ml] def value(
      input: Data,
      coefficients: Coefficients,
      normalizationContext: Broadcast[NormalizationContext]): Double =
    super.value(input, coefficients, normalizationContext) + l2RegValue(convertToVector(coefficients))

  /**
   * Compute the L2 regularization value for the given model coefficients.
   *
   * @param coefficients The model coefficients
   * @return The L2 regularization value
   */
  protected def l2RegValue(coefficients: Vector[Double]): Double =
    l2RegWeight * coefficients.dot(coefficients) / 2
}

/**
 * Trait for a DiffFunction with L2 regularization.
 */
trait L2RegularizationDiff extends DiffFunction with L2Regularization {
  /**
   * Compute the value of the function with L2 regularization over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the objective value
   * @param coefficients The model coefficients used to compute the function's value
   * @param normalizationContext The normalization context
   * @return The computed value of the function
   */
  abstract override protected[ml] def value(
      input: Data,
      coefficients: Coefficients,
      normalizationContext: Broadcast[NormalizationContext]): Double =
    calculate(input, coefficients, normalizationContext)._1

  /**
   * Compute the gradient of the function with L2 regularization over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the gradient
   * @param coefficients The model coefficients used to compute the function's gradient
   * @param normalizationContext The normalization context
   * @return The computed gradient of the function
   */
  abstract override protected[ml] def gradient(
      input: Data,
      coefficients: Coefficients,
      normalizationContext: Broadcast[NormalizationContext]): Vector[Double] =
    calculate(input, coefficients, normalizationContext)._2

  /**
   * Compute both the value and the gradient of the function with L2 regularization for the given model coefficients
   * (computing value and gradient at once is sometimes more efficient than computing them sequentially).
   *
   * @param input The given data over which to compute the value and gradient
   * @param coefficients The model coefficients used to compute the function's value and gradient
   * @param normalizationContext The normalization context
   * @return The computed value and gradient of the function
   */
  abstract override protected[ml] def calculate(
      input: Data,
      coefficients: Coefficients,
      normalizationContext: Broadcast[NormalizationContext]): (Double, Vector[Double]) = {

    val (baseValue, baseGradient) = super.calculate(input, coefficients, normalizationContext)
    val valueWithRegularization = baseValue + l2RegValue(convertToVector(coefficients))
    val gradientWithRegularization = baseGradient + l2RegGradient(convertToVector(coefficients))

    (valueWithRegularization, gradientWithRegularization)
  }

  /**
   * Compute the gradient of the L2 regularization term for the given model coefficients.
   *
   * @param coefficients The model coefficients
   * @return The gradient of the L2 regularization term
   */
  protected def l2RegGradient(coefficients: Vector[Double]): Vector[Double] = coefficients * l2RegWeight
}

/**
 * Trait for a TwiceDiffFunction with L2 regularization.
 */
trait L2RegularizationTwiceDiff extends TwiceDiffFunction with L2RegularizationDiff {
  /**
   * Compute the Hessian of the function with L2 regularization over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the Hessian
   * @param coefficients The model coefficients used to compute the function's hessian, multiplied by a given vector
   * @param multiplyVector The given vector to be dot-multiplied with the Hessian. For example, in conjugate
   *                       gradient method this would correspond to the gradient multiplyVector.
   * @param normalizationContext The normalization context
   * @return The computed Hessian multiplied by the given multiplyVector
   */
  abstract override protected[ml] def hessianVector(
      input: Data,
      coefficients: Coefficients,
      multiplyVector: Coefficients,
      normalizationContext: Broadcast[NormalizationContext]): Vector[Double] =
    super.hessianVector(input, coefficients, multiplyVector, normalizationContext) +
      l2RegHessianVector(convertToVector(multiplyVector))

  /**
   * Compute the diagonal of the Hessian matrix for the function with L2 regularization over the given data for the
   * given model coefficients.
   *
   * @param input The given data over which to compute the diagonal of the Hessian matrix
   * @param coefficients The model coefficients used to compute the diagonal of the Hessian matrix
   * @return The computed diagonal of the Hessian matrix
   */
  abstract override protected[ml] def hessianDiagonal(input: Data, coefficients: Coefficients): Vector[Double] =
    super.hessianDiagonal(input, coefficients) + l2RegHessianDiagonal

  /**
   * Compute the Hessian vector of the L2 regularization term for the given Hessian multiplication vector.
   *
   * @param multiplyVector The Hessian multiplication vector
   * @return The Heassian vector of the L2 regularization term
   */
  protected def l2RegHessianVector(multiplyVector: Vector[Double]): Vector[Double] = multiplyVector * l2RegWeight

  /**
   * Compute the Hessian diagonal of the L2 regularization term.
   *
   * @return The Hessian diagonal of the L2 regularization term
   */
  protected def l2RegHessianDiagonal: Double = l2RegWeight
}
