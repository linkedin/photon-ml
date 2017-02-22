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
package com.linkedin.photon.ml.projector

import breeze.linalg.Vector

/**
 * A trait that performs two types of projections:
 * <ul>
 * <li>
 *   Project the feature vector from the original space to the projected space, usually during model training phase.
 * </li>
 * <li>
 *   Project the coefficients from the projected space back to the original space, usually after model training and
 *   during the model storing nad postprocessing phase.
 * </li>
 * </ul>
 */
protected[ml] trait Projector {

  /**
   * Dimension of the original space
   */
  val originalSpaceDimension: Int

  /**
   * Dimension of the projected space
   */
  val projectedSpaceDimension: Int

  /**
   * Project the feature vector from the original space to the projected space.
   *
   * @param features The input feature vector in the original space
   * @return The feature vector in the projected space
   */
  def projectFeatures(features: Vector[Double]): Vector[Double]

  /**
   * Project the coefficient vector from the projected space back to the original space.
   *
   * @param coefficients The input coefficient vector in the projected space
   * @return The coefficient vector in the original space
   */
  def projectCoefficients(coefficients: Vector[Double]): Vector[Double]
}
