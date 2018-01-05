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

import breeze.linalg.Vector

import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * Trait for twice differentiable function.
 */
trait TwiceDiffFunction extends DiffFunction {
  /**
   * Compute (Hessian * d_i) of the function over the given data for the given model coefficients.
   *
   * @note For more information, see [[http://www.csie.ntu.edu.tw/%7Ecjlin/papers/logistic.pdf]]
   *
   * @param input The given data over which to compute the Hessian
   * @param coefficients The model coefficients used to compute the function's hessian, multiplied by a given vector
   * @param multiplyVector The given vector to be dot-multiplied with the Hessian. For example, in conjugate
   *                       gradient method this would correspond to the gradient multiplyVector.
   * @param normalizationContext The normalization context
   * @return The computed Hessian multiplied by the given multiplyVector
   */
  protected[ml] def hessianVector(
    input: Data,
    coefficients: Coefficients,
    multiplyVector: Coefficients,
    normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double]

  /**
   * Compute the diagonal of Hessian matrix of the function over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the diagonal of the Hessian matrix
   * @param coefficients The model coefficients used to compute the diagonal of the Hessian matrix
   * @return The computed diagonal of the Hessian matrix
   */
  protected[ml] def hessianDiagonal(input: Data, coefficients: Coefficients): Vector[Double]
}
