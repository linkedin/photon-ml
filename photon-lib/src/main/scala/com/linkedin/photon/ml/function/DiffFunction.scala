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
 * Trait for a differentiable function.
 */
trait DiffFunction extends ObjectiveFunction {

  /**
   * Compute the gradient of the function over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the gradient
   * @param coefficients The model coefficients used to compute the function's gradient
   * @param normalizationContext The normalization context
   * @return The computed gradient of the function
   */
  protected[ml] def gradient(
    input: Data,
    coefficients: Vector[Double],
    normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double]

  /**
   * Compute both the value and the gradient of the function for the given model coefficients (computing value and
   * gradient at once is sometimes more efficient than computing them sequentially).
   *
   * @param input The given data over which to compute the value and gradient
   * @param coefficients The model coefficients used to compute the function's value and gradient
   * @param normalizationContext The normalization context
   * @return The computed value and gradient of the function
   */
  protected[ml] def calculate(
    input: Data,
    coefficients: Vector[Double],
    normalizationContext: BroadcastWrapper[NormalizationContext]): (Double, Vector[Double])
}
