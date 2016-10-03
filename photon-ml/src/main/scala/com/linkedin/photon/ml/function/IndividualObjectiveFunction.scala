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

import com.linkedin.photon.ml.data.LabeledPoint

/**
 * The base objective function used by IndividualOptimizationProblems. This function works with data locally as part of
 * a single task (on a single executor).
 */
abstract class IndividualObjectiveFunction extends ObjectiveFunction with Serializable {
  type Data = Iterable[LabeledPoint]
  type Coefficients = Vector[Double]

  /**
   * Compute the size of the domain for the given input data (i.e. the number of features, including the intercept if
   * there is one).
   *
   * @param input The given data for which to compute the domain dimension
   * @return The domain dimension
   */
  override protected[ml] def domainDimension(input: Iterable[LabeledPoint]): Int = input.head.features.size

  /**
   * IndividualOptimizationProblems compute objective value over all of the data at once as part of a single task (on
   * a single executor). Thus, the IndividualObjectiveFunction handles Vectors directly.
   *
   * @param coefficients A coefficients Vector to convert
   * @return The given coefficients Vector
   */
  override protected[ml] def convertFromVector(coefficients: Vector[Double]): Coefficients = coefficients

  /**
   * IndividualOptimizationProblems handle Vectors directly, so the IndividualObjectiveFunction input is already a
   * Vector.
   *
   * @param coefficients A Coefficients object to convert
   * @return The given coefficients Vector
   */
  override protected[ml] def convertToVector(coefficients: Vector[Double]): Vector[Double] = coefficients
}
