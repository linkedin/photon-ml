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
package com.linkedin.photon.ml.function.glm

/**
 * The l(z, y) function for a generalized linear model, where:
 *
 *  - z = theta^T^x + offset
 *  - theta is the coefficient vector
 *  - x is the feature vector
 *  - y is the label
 *
 * l(z, y) is the negative of the log-likelihood (discounting the irrelevant constant term).
 *
 * For example:
 *
 *    l(z, y) = 1/2 (z - y)^2^.
 *
 * for linear regression.
 *
 * Supports computing the value, first derivative, and second derivative at a single point.
 *
 * @note Function names follow the differentiation notation found here:
 *       [[http://www.wikiwand.com/en/Notation_for_differentiation#/Euler.27s_notation]]
 */
trait PointwiseLossFunction extends Serializable {

  /**
   * Calculate the loss function value and 1st derivative with respect to z.
   *
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The value and the 1st derivative with respect to z
   */
  def lossAndDzLoss(margin: Double, label: Double): (Double, Double)

  /**
   * Calculate the 2nd derivative with respect to z.
   *
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The 2nd derivative with respect to z
   */
  def DzzLoss(margin: Double, label: Double): Double
}
