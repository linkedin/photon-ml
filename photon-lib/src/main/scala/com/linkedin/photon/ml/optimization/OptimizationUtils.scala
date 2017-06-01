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
package com.linkedin.photon.ml.optimization

import breeze.linalg.Vector

import com.linkedin.photon.ml.util.VectorUtils

/**
 * General purpose utils that can be leveraged by the various optimizers.
 */
object OptimizationUtils {
  /**
   * Project a coefficient to a constrained space, specified by a bounding range [L, U]:
   *
   *  L <= coefficient <= U
   *
   * @param coefficient The value to be projected
   * @param bounds The bounding range
   * @return New value of coeff after projection
   */
  private[this] def projectCoefficientToInterval(coefficient: Double, bounds: Option[(Double, Double)]): Double =
    bounds match {
      case Some((lowerBound, upperBound)) =>
        if (coefficient < lowerBound) {
          lowerBound
        } else if (coefficient > upperBound) {
          upperBound
        } else {
          coefficient
        }

      case None => coefficient
    }

  /**
   * Return new coefficients after projection into the constrained space as specified by the map of feature index to
   * the (lowerBound, upperBound) constraints.
   *
   * @param coefficients Coefficients to be projected
   * @param constraintMapOption Map of feature index to the bounds that are to be enforced on that particular feature
   * @return Projected value of coefficients
   */
  def projectCoefficientsToSubspace(
      coefficients: Vector[Double],
      constraintMapOption: Option[Map[Int, (Double, Double)]]): Vector[Double] =

    constraintMapOption match {
      case Some(constraintMap) =>
        val projectedCoefficients = VectorUtils.zeroOfSameType(coefficients)
        coefficients.activeIterator.foreach { case (key, value) =>
          projectedCoefficients.update(key, projectCoefficientToInterval(value, constraintMap.get(key)))
        }

        projectedCoefficients

      case None => coefficients
    }
}
