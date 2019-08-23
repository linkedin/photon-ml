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

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper
import com.linkedin.photon.ml.model.{Coefficients => ModelCoefficients}

trait PriorDistribution extends ObjectiveFunction {

  protected var _previousCoefficients: ModelCoefficients = _
  protected var _l1RegWeight: Double = 0D
  protected var _l2RegWeight: Double = 0D
  // Todo: change prevHessian type to Matrix[Double] later
  protected var prevMeans: Vector[Double] = _
  protected var prevVariances: Vector[Double] = _
  protected var prevHessian: Vector[Double] = _

  /**
   * Getter.
   *
   * @return The L1 regularization weight
   */
  def getL1RegularizationWeight: Double = _l1RegWeight

  /**
   * Getter.
   *
   * @return The L2 regularization weight
   */
  def getL2RegularizationWeight: Double = _l2RegWeight

  /**
   * Getter.
   *
   * @return The previous coefficients
   */
  def getPreviousCoefficients: ModelCoefficients = _previousCoefficients


  /**
   * Setter.
   *
   * @note This function definition uses the setter syntactic sugar trick. Statements like:
   *
   *    objectiveFunction.l1RegularizationWeight = 10
   *
   *       will call this function.
   * @param newL1RegWeight The new L1 regularization weight
   */
  protected[ml] def l1RegularizationWeight_=(newL1RegWeight: Double): Unit = _l1RegWeight = newL1RegWeight

  /**
   * Setter.
   *
   * @note This function definition uses the setter syntactic sugar trick. Statements like:
   *
   *    objectiveFunction.l2RegularizationWeight = 10
   *
   *       will call this function.
   * @param newL2RegWeight The new L2 regularization weight
   */
  protected[ml] def l2RegularizationWeight_=(newL2RegWeight: Double): Unit = _l2RegWeight = newL2RegWeight

  /**
   * Setter.
   *
   * @note This function definition uses the setter syntactic sugar trick. Statements like:
   *
   *    objectiveFunction.previousCoefficients_ = newPreviousCoefficients
   *
   *       will call this function.
   * @param newPreviousCoefficients The new previous coefficients
   */
  protected[ml] def previousCoefficients_=(newPreviousCoefficients: ModelCoefficients): Unit  = {

    _previousCoefficients = newPreviousCoefficients
    val priorDistributionStats = _previousCoefficients match {
      case ModelCoefficients(means: Vector[Double], Some(variances: Vector[Double])) => (means, variances)
      case ModelCoefficients(means: Vector[Double], None) =>
        throw new IllegalArgumentException("Previous variances should not be empty.")
    }
    prevMeans = priorDistributionStats._1
    prevVariances = priorDistributionStats._2
    // Todo: update previous hessian's initialization method
    prevHessian = prevVariances.map(v => 1D / math.max(v, MathConst.EPSILON))
//    updatePriorDistribution()
  }

//  private[ml] def updatePriorDistribution(): Unit = {
//    (_prevMeans, _prevVariances) = _previousCoefficients match {
//      case ModelCoefficients(means: Vector[Double], Some(variances: Vector[Double])) => (means, variances)
//      case ModelCoefficients(means: Vector[Double], None) =>
//        throw new IllegalArgumentException("Previous variances cannot be empty.")
//      case _ =>
//        throw new IllegalArgumentException("Previous means cannot be empty.")
//    }
//    //    prevMeans = priorDistributionStats._1
//    //    prevVariances = priorDistributionStats._2
//    // Todo: update previous hessian's initialization method
//    _prevHessian = _prevVariances.map(v => 1D / math.max(v, MathConst.EPSILON))
//  }

  /**
   * Compute the value of the function with L1 and L2 regularization over the given data for the given model
   * coefficients.
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
    super.value(input, coefficients, normalizationContext) + l1RegValue(convertToVector(coefficients)) +
      l2RegValue(convertToVector(coefficients))

  /**
   * Compute the L1 regularization value for the given model coefficients.
   *
   * @param coefficients The model coefficients
   * @return The L1 regularization value
   */
  protected def l1RegValue(coefficients: Vector[Double]): Double = {

    val normalizedCoefficients = (coefficients - prevMeans) :/ sqrt(prevVariances)
    _l1RegWeight * sum(abs(normalizedCoefficients))
  }

  /**
   * Compute the L2 regularization value for the given model coefficients.
   *
   * @param coefficients The model coefficients
   * @return The L2 regularization value
   */
  protected def l2RegValue(coefficients: Vector[Double]): Double = {

    val normalizedCoefficients = (coefficients - prevMeans) :/ sqrt(prevVariances)
    _l2RegWeight * normalizedCoefficients.dot(normalizedCoefficients) / 2
  }
}

trait PriorDistributionDiff extends DiffFunction with PriorDistribution {

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
    val valueWithRegularization = baseValue + l1RegValue(convertToVector(coefficients)) +
      l2RegValue(convertToVector(coefficients))
    val gradientWithRegularization = baseGradient + l1RegGradient(convertToVector(coefficients)) +
      l2RegGradient(convertToVector(coefficients))

    (valueWithRegularization, gradientWithRegularization)
  }

  /**
   * Compute the gradient of the L1 regularization term for the given model coefficients.
   *
   * @param coefficients The model coefficients
   * @return The gradient of the L1 regularization term
   */
  protected def l1RegGradient(coefficients: Vector[Double]): Vector[Double] = {

    // Todo: what if normalizedCoefficient[i] == 0?
    val coefficientsMask = (coefficients - prevMeans).map(coefficient => if (coefficient > 0) 1.0 else -1.0)
    _l1RegWeight * (coefficientsMask :/ sqrt(prevVariances))
  }

  /**
   * Compute the gradient of the L2 regularization term for the given model coefficients.
   *
   * @param coefficients The model coefficients
   * @return The gradient of the L2 regularization term
   */
  protected def l2RegGradient(coefficients: Vector[Double]): Vector[Double] = {

    val normalizedCoefficients = (coefficients - prevMeans) :/ sqrt(prevVariances)
    _l2RegWeight * normalizedCoefficients
  }
}

trait PriorDistributionTwiceDiff extends TwiceDiffFunction with PriorDistributionDiff {
  /**
   * Compute the Hessian of the function with L2 regularization over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the Hessian
   * @param coefficients The model coefficients used to compute the function's hessian, multiplied by a given vector
   * @param multiplyVector The given vector to be dot-multiplied with the Hessian. For example, in conjugate
   * gradient method this would correspond to the gradient multiplyVector.
   * @param normalizationContext The normalization context
   * @return The computed Hessian multiplied by the given multiplyVector
   */
  abstract override protected[ml] def hessianVector(
      input: Data,
      coefficients: Coefficients,
      multiplyVector: Coefficients,
      normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
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
    super.hessianDiagonal(input, coefficients) :+ l2RegHessianDiagonal

  /**
   * Compute the Hessian matrix for the function with L2 regularization over the given data for the given model
   * coefficients.
   *
   * @param input The given data over which to compute the diagonal of the Hessian matrix
   * @param coefficients The model coefficients used to compute the diagonal of the Hessian matrix
   * @return The computed Hessian matrix
   */
  abstract override protected[ml] def hessianMatrix(input: Data, coefficients: Coefficients): DenseMatrix[Double] = {

    // Todo: to be implemented with previous variances
    val hessianMatrix = super.hessianMatrix(input, coefficients)
//    + diag(DenseVector(l2RegHessianDiagonal))
    hessianMatrix
  }

  /**
   * Compute the Hessian vector of the L2 regularization term for the given Hessian multiplication vector.
   *
   * @param multiplyVector The Hessian multiplication vector
   * @return The Heassian vector of the L2 regularization term
   */
  protected def l2RegHessianVector(multiplyVector: Vector[Double]): Vector[Double] =
    _l2RegWeight * (multiplyVector :/ sqrt(prevVariances))

  /**
   * Compute the Hessian diagonal of the L2 regularization term.
   *
   * @return The Hessian diagonal of the L2 regularization term
   */
  protected def l2RegHessianDiagonal: Vector[Double] = _l2RegWeight / prevVariances
}
