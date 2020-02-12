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
package com.linkedin.photon.ml.model

import breeze.linalg.{Vector, norm}
import breeze.stats.meanAndVariance

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.util.{MathUtils, Summarizable, VectorUtils}

/**
 * Coefficients are a wrapper to store means and variances of model coefficients together.
 *
 * @note Breeze's SparseVector does NOT sort the non-zeros in order of increasing index, but still supports a get/set
 *       backed by a binary search (11/18/2016)
 * @param means The mean of the model coefficients
 * @param variancesOption Optional variance of the model coefficients
 */
case class Coefficients(means: Vector[Double], variancesOption: Option[Vector[Double]] = None)
  extends Summarizable {

  // Force means and variances to be of the same type (dense or sparse). This seems reasonable
  // and greatly reduces the number of combinations to check in unit testing.
  require(variancesOption.isEmpty || variancesOption.get.getClass == means.getClass,
    "Coefficients: If variances are provided, must be of the same vector type as means")
  // GAME over if variances are given but don't have the same length as the vector of means
  require(variancesOption.isEmpty || variancesOption.get.length == means.length,
    "Coefficients: Means and variances have different lengths")

  def length: Int = means.length
  lazy val meansL2Norm: Double = norm(means, 2)
  lazy val variancesL2NormOption: Option[Double] = variancesOption.map(variances => norm(variances, 2))

  /**
   * Compute the score for the given features.
   *
   * @note Score can be done with either sparse or dense vectors for the features.
   * @param features Features to score
   * @return The score
   */
  def computeScore(features: Vector[Double]): Double = {
    require(
      means.length == features.length,
      s"Coefficients length (${means.length}) != features length (${features.length})")

    means.dot(features)
  }

  /**
   * Output a human-friendly string describing this Coefficients vector.
   *
   * @return A summary of the object in string representation
   */
  override def toSummaryString: String = {
    val sb = new StringBuilder()
    val isDense = means.getClass.getName.contains("Dense")
    val meanAndVar = meanAndVariance(means)

    sb.append(s"Type: ${if (isDense) "dense" else "sparse"}, ")
    sb.append(s"length: $length, ${if (variancesOption.isDefined) "with variances" else "without variances" }\n")
    if (!isDense) {
      sb.append(s"Number of declared non-zeros: ${means.activeSize}\n")
      sb.append(
        s"Number of counted non-zeros (abs > ${MathConst.EPSILON}): ${means.toArray.count(!MathUtils.isAlmostZero(_))}")
      sb.append("\n")
    }
    sb.append(s"Mean and stddev of the mean: ${meanAndVar.mean} ${meanAndVar.stdDev}\n")
    sb.append(s"l2 norm of the mean: $meansL2Norm\n")
    variancesL2NormOption.map(norm => sb.append(s"l2 norm of the variance $norm"))

    sb.toString()
  }

  /**
   * Returns a string representation of the [[Coefficients]]
   *
   * @return A string representation of the coefficients vector
   */
  override def toString: String = means.toString

  /**
   * Equality of coefficients is only within some tolerance. Also, Breeze's Vector can be either a dense or a sparse
   * vector, and we require that those match for both arguments of this equality.
   * Also note that in Breeze, a SparseVector and a DenseVector are equal if they contain the same values at the same
   * indexes, i.e. the types "SparseVector" and "DenseVector" do not matter for equality in Breeze. Here we want to
   * be stricter and declare inequality if the 2 Coefficients instances use different sub-types of Vector.
   *
   * @param that The other Coefficients to compare to
   * @return True if the Coefficients are equal, false otherwise
   */
  override def equals(that: Any): Boolean =
    that match {
      case other: Coefficients =>
        val (m1, v1, m2, v2) = (this.means, this.variancesOption, other.means, other.variancesOption)
        val sameType = m1.getClass == m2.getClass && v1.map(_.getClass) == v2.map(_.getClass)
        lazy val sameMeans = VectorUtils.areAlmostEqual(m1, m2)
        lazy val sameVariance = (v1, v2) match {
          case (None, None) => true
          case (Some(val1), Some(val2)) => VectorUtils.areAlmostEqual(val1, val2)
          case (_, _) => false
        }

        sameType && sameMeans && sameVariance

      case _ => false
    }

  /**
   * Returns a hash code value for the object.
   *
   * TODO: Violation of the hashCode() contract
   *
   * @return An [[Int]] hash code
   */
  override def hashCode(): Int = super.hashCode()
}

protected[ml] object Coefficients {

  /**
   * Create a zero coefficient vector.
   *
   * @param dimension Dimensionality of the coefficient vector
   * @return Zero coefficient vector
   */
  def initializeZeroCoefficients(dimension: Int): Coefficients = {
    Coefficients(Vector.zeros[Double](dimension), variancesOption = None)
  }
}
