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
package com.linkedin.photon.ml.model

import breeze.linalg.{DenseVector, SparseVector, Vector, norm}
import breeze.stats.meanAndVariance

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.util.Summarizable
import com.linkedin.photon.ml.util.VectorUtils

/**
 * Coefficients are a wrapper to store means and variances of model coefficients together.
 *
 * NOTE: Breeze's SparseVector does NOT sort the non-zeros in order of increasing index, but still supports
 * a get/set backed by a binary search!!! (11/18/2016)
 *
 * @param means The mean of the model coefficients
 * @param variancesOption Optional variance of the model coefficients
 */
protected[ml] case class Coefficients(means: Vector[Double], variancesOption: Option[Vector[Double]] = None)
  extends Summarizable {

  // Force means and variances to be of the same type (dense or sparse). This seems reasonable
  // and greatly reduces the number of combinations to check in unit testing.
  require(variancesOption.isEmpty || variancesOption.get.getClass == means.getClass,
    "Coefficients: If variances are provided, must be of the same vector type as means")
  // Game over if variances are given but don't have the same length as the vector of means
  require(variancesOption.isEmpty || variancesOption.get.length == means.length,
    "Coefficients: Means and variances have different lengths")

  def length: Int = means.length
  lazy val meansL2Norm: Double = norm(means, 2)
  lazy val variancesL2NormOption: Option[Double] = variancesOption.map(variances => norm(variances, 2))

  /**
   * Compute the score for the given features.
   *
   * @note Score can be done with either sparse or dense vectors for the features.
   *
   * @param features Features to score
   * @return The score
   */
  def computeScore(features: Vector[Double]): Double = {
    require(means.length == features.length,
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
    val eps = MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD
    sb.append(s"Type: ${if (isDense) "dense" else "sparse"}, ")
    sb.append(s"length: $length, ${if (variancesOption.isDefined) "with variances" else "without variances" }\n")
    if (!isDense) {
      sb.append(s"Number of declared non-zeros: ${means.activeSize}\n")
      sb.append(s"Number of counted non-zeros (abs > $eps): ${means.toArray.count(Math.abs(_) > eps)}")
      sb.append("\n")
    }
    val meanAndVar = meanAndVariance(means)
    sb.append(s"Mean and stddev of the mean: ${meanAndVar.mean} ${meanAndVar.stdDev}\n")
    sb.append(s"l2 norm of the mean: $meansL2Norm\n")
    variancesL2NormOption.map(_ => sb.append(s"l2 norm of the variance ${variancesL2NormOption.get}"))
    sb.toString()
  }

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
  override def equals(that: Any): Boolean = {

    that match {
      case other: Coefficients =>
        val (m1, v1, m2, v2) = (this.means, this.variancesOption, other.means, other.variancesOption)
        val sameType = m1.getClass == m2.getClass && v1.map(_.getClass) == v2.map(_.getClass)
        lazy val sameMeans = m1.length == m2.length && VectorUtils.areAlmostEqual(m1, m2)
        lazy val sameVariance = (v1, v2) match {
          case (None, None) => true
          case (Some(val1), Some(val2)) => val1.length == val2.length && VectorUtils.areAlmostEqual(val1, val2)
          case (_, _) => false
        }

        sameType && sameMeans && sameVariance

      case _ => false
    }
  }

  /**
   *
   * @return
   */
  // TODO: Violation of the hashCode() contract
  override def hashCode(): Int = {
    super.hashCode()
  }

  /**
   *
   * @return
   */
  override def toString: String = means.toString
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

  /**
   * Constructor for an instance backed by a Breeze DenseVector.
   *
   * @param values The coefficients values
   * @return An instance of Coefficients
   */
  def apply(values: Double*): Coefficients =
    Coefficients(new DenseVector[Double](Array[Double](values: _*)))

  /**
   * Constructor for an instance backed by a Breeze SparseVector.
   *
   * @note The non-zeros must be sorted in order of increasing indices!!!
   *
   * @param length The Coefficients dimension ( (1,0,0,4) has dimension 4)
   * @param indices The indices of the non-zeros
   * @param nnz The non-zero values
   * @return An instance of Coefficients
   */
  def apply(length: Int)(indices: Int*)(nnz: Double*): Coefficients = {

    require(0 < length)
    require(indices.length == nnz.length)
    require(indices.sorted == indices)
    // TODO: check for duplicates?

    Coefficients(new SparseVector[Double](Array[Int](indices: _*), Array[Double](nnz: _*), length))
  }
}
