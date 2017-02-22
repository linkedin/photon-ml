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
 * The base objective function class for an optimization problem.
 */
abstract class ObjectiveFunction {
  type Data
  type Coefficients

  /**
   * Compute the value of the function over the given data for the given model coefficients.
   *
   * @param input The given data over which to compute the objective value
   * @param coefficients The model coefficients used to compute the function's value
   * @param normalizationContext The normalization context
   * @return The computed value of the function
   */
  protected[ml] def value(
      input: Data,
      coefficients: Coefficients,
      normalizationContext: Broadcast[NormalizationContext]): Double

  /**
   * Compute the size of the domain for the given input data (i.e. the number of features, including the intercept if
   * there is one).
   *
   * @param input The given data for which to compute the domain dimension
   * @return The domain dimension
   */
  protected[ml] def domainDimension(input: Data): Int

  /**
   * Convert a coefficients Vector to an object of the type used by the function.
   *
   * @param coefficients A coefficients Vector to convert
   * @return The given coefficients Vector as an object of the type expected by the function
   */
  protected[ml] def convertFromVector(coefficients: Vector[Double]): Coefficients

  /**
   * Convert a coefficients object of the type used by the function to a Vector.
   *
   * @param coefficients A coefficients object to convert
   * @return The given coefficients object as a Vector
   */
  protected[ml] def convertToVector(coefficients: Coefficients): Vector[Double]

  /**
   * Cleanup the coefficients object created by a call to convertFromVector, if necessary.
   *
   * @param coefficients A coefficients object to cleanup
   */
  protected[ml] def cleanupCoefficients(coefficients: Coefficients): Unit = {}
}
