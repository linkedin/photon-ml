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
package com.linkedin.photon.ml.function

import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, Vector, diag}
import scala.math.pow

import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * Trait for an objective function with L2 regularization.
 */
trait L2Regularization extends ObjectiveFunction {

  protected var l2RegWeight: Double = 0D

  require(interceptOpt.forall(_ >= 0), "Intercept index is negative.")

  def interceptOpt: Option[Int] = None

  /**
   * Getter.
   *
   * @return The L2 regularization weight
   */
  def l2RegularizationWeight: Double = l2RegWeight

  /**
   * Setter.
   *
   * @note This function definition uses the setter syntactic sugar trick. Statements like:
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
      normalizationContext: BroadcastWrapper[NormalizationContext]): Double =
    super.value(input, coefficients, normalizationContext) + l2RegValue(convertToVector(coefficients))

  /**
   * Compute the L2 regularization value for the given model coefficients.
   *
   * @param coefficients The model coefficients
   * @return The L2 regularization value
   */
  protected def l2RegValue(coefficients: Vector[Double]): Double = {

    val coefSquaredMagnitude = coefficients.dot(coefficients)

    val unweightedL2Regularization = interceptOpt match {
      case None =>
        coefSquaredMagnitude

      case Some(interceptIndex) =>
        val coefLength = coefficients.length

        require(interceptIndex < coefLength, s"Intercept index out of bounds: $interceptIndex >= $coefLength")

        coefSquaredMagnitude - pow(coefficients(interceptIndex), 2)
    }

    l2RegWeight * unweightedL2Regularization / 2
  }
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
      normalizationContext: BroadcastWrapper[NormalizationContext]): Double =
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
      normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
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
      normalizationContext: BroadcastWrapper[NormalizationContext]): (Double, Vector[Double]) = {

    val (baseValue, baseGradient) = super.calculate(input, coefficients, normalizationContext)
    val valueWithRegularization = baseValue + l2RegValue(convertToVector(coefficients))
    val gradientWithRegularization = baseGradient +:+ l2RegGradient(convertToVector(coefficients))

    (valueWithRegularization, gradientWithRegularization)
  }

  /**
   * Compute the gradient of the L2 regularization term for the given model coefficients.
   *
   * @param coefficients The model coefficients
   * @return The gradient of the L2 regularization term
   */
  protected def l2RegGradient(coefficients: Vector[Double]): Vector[Double] = {

    val l2Gradient = coefficients * l2RegWeight

    interceptOpt match {
      case Some(interceptIndex) =>
        l2Gradient -= SparseVector[Double](coefficients.length)((interceptIndex, l2Gradient(interceptIndex)))

      case None =>
    }

    l2Gradient
  }
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
      normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
    super.hessianVector(input, coefficients, multiplyVector, normalizationContext) +:+
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
    super.hessianDiagonal(input, coefficients) +:+ l2RegHessianDiagonal(convertToVector(coefficients).length)

  /**
   * Compute the Hessian matrix for the function with L2 regularization over the given data for the given model
   * coefficients.
   *
   * @param input The given data over which to compute the diagonal of the Hessian matrix
   * @param coefficients The model coefficients used to compute the diagonal of the Hessian matrix
   * @return The computed Hessian matrix
   */
  abstract override protected[ml] def hessianMatrix(input: Data, coefficients: Coefficients): DenseMatrix[Double] =
    super.hessianMatrix(input, coefficients) +:+ l2RegHessianMatrix(convertToVector(coefficients).length)

  /**
   * Compute the Hessian vector of the L2 regularization term for the given Hessian multiplication vector.
   *
   * @param multiplyVector The Hessian multiplication vector
   * @return The Hessian vector of the L2 regularization term
   */
  protected def l2RegHessianVector(multiplyVector: Vector[Double]): Vector[Double] = {

    val l2HessianVector = multiplyVector * l2RegWeight

    interceptOpt match {
      case Some(interceptIndex) =>
        l2HessianVector -= SparseVector[Double](multiplyVector.length)((interceptIndex, l2HessianVector(interceptIndex)))

      case None =>
    }

    l2HessianVector
  }

  /**
   * Compute the Hessian diagonal of the L2 regularization term.
   *
   * @param dimension The dimension of the Hessian matrix (only one number since Hessian matrix is square)
   * @return The Hessian diagonal of the L2 regularization term
   */
  protected def l2RegHessianDiagonal(dimension: Int): DenseVector[Double] = {

    val l2HessianDiagonal = DenseVector.fill(dimension, l2RegWeight)

    interceptOpt match {
      case Some(interceptIndex) =>
        l2HessianDiagonal -= SparseVector[Double](dimension)((interceptIndex, l2RegWeight))

      case None =>
    }

    l2HessianDiagonal
  }

  /**
   * Compute the Hessian matrix of the L2 regularization term.
   *
   * @param dimension The dimension of the Hessian matrix (only one number since Hessian matrix is square)
   * @return The Hessian matrix of the L2 regularization term
   */
  protected def l2RegHessianMatrix(dimension: Int): DenseMatrix[Double] = diag(l2RegHessianDiagonal(dimension))
}
