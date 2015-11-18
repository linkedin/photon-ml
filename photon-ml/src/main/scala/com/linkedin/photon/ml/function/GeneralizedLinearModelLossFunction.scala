/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
import com.linkedin.photon.ml.data.{ObjectProvider, LabeledPoint}
import com.linkedin.photon.ml.normalization.NormalizationContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD


/**
 * This class is used to calculate value, gradient and Hessian of generalized linear models.
 * The loss function of a generalized linear model can all be expressed as
 *
 * L(w) = \sum_i l(z_i, y_i)
 *
 * with z_i = w^T^ z_i.
 *
 * Different generalized linear models will have different l(z, y). The functionality of l(z, y) is provided by
 * [[PointwiseLossFunction]]. Since the loss function could be changed in different normalization type,
 * a normalization context object is used to indicate the normalization strategy to evaluate this loss function.
 *
 * All generalized linear model loss function should inherite from this class.
 *
 * TODO: Note that calculateAt and hessianVectorAt methods has been annotated as deprecated and will throw an exception if
 * TODO  called. The segregation of calculateAt and hessianVectorAt with calculate and hessianVector has happen in
 * TODO  DiffFunction.withRegularization and TwiceDiffFunction.withRegularization. This will be fixed at the redesign
 * TODO  ticket: https://jira01.corp.linkedin.com:8443/browse/OFFREL-489
 *
 * @param singleLossFunction A single loss function l(z, y) used for the generalized linear model
 * @param normalizationContext The normalization context used to calculate this loss function
 *
 * @author dpeng
 */
class GeneralizedLinearModelLossFunction(singleLossFunction: PointwiseLossFunction, normalizationContext: ObjectProvider[NormalizationContext]) extends TwiceDiffFunction[LabeledPoint] {


  /**
   * Compute the Hessian of the function under the given data and coefficients in the normalization context, then multiply it with a given multiplyVector.
   * @param data The given data at which point to compute the hessian multiplied by a given vector
   * @param coefficients The given model coefficients used to compute the hessian multiplied by a given vector
   * @param multiplyVector The given multiplyVector to be multiplied with the Hessian. For example, in conjugate gradient method
   *                       this multiplyVector would correspond to the gradient multiplyVector.
   * @return The computed Hessian multiplied by the given multiplyVector
   */
  override protected[ml] def hessianVector(data: Iterable[LabeledPoint],
                                              coefficients: Vector[Double],
                                              multiplyVector: Vector[Double]): Vector[Double] = {
    HessianVectorAggregator.calcHessianVector(data, coefficients, multiplyVector, singleLossFunction, normalizationContext)
  }

  /**
   * Compute the Hessian of the function under the given data and coefficients in the normalization context, then multiply it with a given multiplyVector.
   * @param data The given data at which point to compute the hessian multiplied by a given vector
   * @param broadcastedCoefficients The broadcasted model coefficients used to compute the hessian multiplied by a given vector
   * @param multiplyVector The given multiplyVector to be multiplied with the Hessian. For example, in conjugate gradient method
   *                       this multiplyVector would correspond to the gradient multiplyVector.
   * @return The computed Hessian multiplied by the given multiplyVector
   */
  override protected[ml] def hessianVector(data: RDD[LabeledPoint],
                                              broadcastedCoefficients: Broadcast[Vector[Double]],
                                              multiplyVector: Broadcast[Vector[Double]]): Vector[Double] = {
    HessianVectorAggregator.calcHessianVector(data, broadcastedCoefficients, multiplyVector, singleLossFunction, normalizationContext)
  }

  /**
   * Calculate both the value and the gradient of the function given a RDD of data points and model coefficients
   * in the normalization context (compute value and gradient at once is sometimes more efficient than computing them sequentially)
   * @param data The given data at which point to compute the function's value and gradient
   * @param broadcastedCoefficients The broadcasted model coefficients used to compute the function's value and gradient
   * @return The computed value and gradient of the function
   */
  override protected[ml] def calculate(data: RDD[LabeledPoint],
                                          broadcastedCoefficients: Broadcast[Vector[Double]]): (Double, Vector[Double]) = {
    ValueAndGradientAggregator.calculateValueAndGradient(data, broadcastedCoefficients, singleLossFunction, normalizationContext)
  }

  /**
   * Calculate both the value and the gradient of the function given a sequence of data points and model coefficients
   * in the normalization context (compute value and gradient at once is sometimes more efficient than computing them sequentially)
   * @param data The given data at which point to compute the function's value and gradient
   * @param coefficients The given model coefficients used to compute the function's value and gradient
   * @return The computed value and gradient of the function
   */
  override protected[ml] def calculate(data: Iterable[LabeledPoint],
                                          coefficients: Vector[Double]): (Double, Vector[Double]) = {
    ValueAndGradientAggregator.calculateValueAndGradient(data, coefficients, singleLossFunction, normalizationContext)
  }

  /**
   * This method is not supposed to be called. hessianVector logic does not involves this method.
   * This legacy method is implemented just to comply TwiceDiffFunction trait.
   */
  @Deprecated
  override protected[ml] def hessianVectorAt(datum: LabeledPoint,
                                                coefficients: Vector[Double],
                                                multiplyVector: Vector[Double],
                                                cumHessianVector: Vector[Double]): Unit = {
    throw new UnsupportedOperationException("Do not call GeneralizedLinearModelLossFunction.hessianVectorAt")
  }

  /**
   * This method is not supposed to be called. calculate logic does not involves this method.
   * This legacy method is implemented just to comply DiffFunction trait.
   */
  @Deprecated
  override protected[ml] def calculateAt(datum: LabeledPoint,
                                            coefficients: Vector[Double],
                                            cumGradient: Vector[Double]): Double = {
    throw new UnsupportedOperationException("Do not call GeneralizedLinearModelLossFunction.calculateAt")
  }
}


