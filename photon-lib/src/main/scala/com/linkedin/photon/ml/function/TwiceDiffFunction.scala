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

import breeze.linalg.{DenseMatrix, Vector}

import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * Trait for twice differentiable function.
 */
trait TwiceDiffFunction extends DiffFunction {

  /**
   * Compute the Hessian matrix over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the diagonal of the Hessian matrix
   * @param coefficients The model coefficients used to compute the diagonal of the Hessian matrix
   * @return The computed Hessian matrix
   */
  protected[ml] def hessianMatrix(input: Data, coefficients: Vector[Double]): DenseMatrix[Double]

  /**
   * Compute the diagonal of the Hessian matrix over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the diagonal of the Hessian matrix
   * @param coefficients The model coefficients used to compute the diagonal of the Hessian matrix
   * @return The computed diagonal of the Hessian matrix
   */
  protected[ml] def hessianDiagonal(input: Data, coefficients: Vector[Double]): Vector[Double]

  /**
   * Compute H * d (where H is the Hessian matrix and d is some vector) over the given data for the given model
   * coefficients. This is a special helper function which computes H * d more efficiently than computing the entire
   * Hessian matrix and then multiplying it with d.
   *
   * @note For more information, see equation 7 and algorithm 2 of
   *       [[http://www.csie.ntu.edu.tw/%7Ecjlin/papers/logistic.pdf]].
   *
   * @param input The given data over which to compute the Hessian matrix
   * @param coefficients The model coefficients used to compute the Hessian matrix
   * @param multiplyVector The vector d to be dot-multiplied with the Hessian matrix (e.g. for a conjugate
   *                       gradient method this would correspond to the Newton gradient direction)
   * @param normalizationContext The normalization context
   * @return The vector d multiplied by the Hessian matrix
   */
  protected[ml] def hessianVector(
      input: Data,
      coefficients: Vector[Double],
      multiplyVector: Vector[Double],
      normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double]
}
