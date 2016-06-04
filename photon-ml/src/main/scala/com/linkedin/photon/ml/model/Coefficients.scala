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

import breeze.linalg.{Vector, norm}
import breeze.stats.meanAndVariance
import com.linkedin.photon.ml.util.Summarizable

/**
  * Representation of model coefficients
  *
  * @param means The mean of the model coefficients
  * @param variancesOption Optional variance of the model coefficients
  */
protected[ml] case class Coefficients(means: Vector[Double], variancesOption: Option[Vector[Double]] = None)
  extends Summarizable {

  lazy val meansL2Norm: Double = norm(means, 2)
  lazy val variancesL2NormOption: Option[Double] = variancesOption.map(variances => norm(variances, 2))

  /**
    * Compute the score for the given features
    *
    * @param features Features to score
    * @return The score
    */
  def computeScore(features: Vector[Double]): Double = {
    require(means.length == features.length, s"Coefficients length (${means.length}) != features length (${features.length})")
    means.dot(features)
  }

  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder()
    stringBuilder.append(s"meanAndVarianceAndCount of the mean: ${meanAndVariance(means)}")
    stringBuilder.append(s"\nl2 norm of the mean: ${norm(means, 2)}")
    variancesOption.foreach(variances => s"\nmeanAndVarianceAndCount of variance: ${meanAndVariance(variances)}")
    stringBuilder.toString()
  }

  override def equals(that: Any): Boolean = {
    that match {
      case other: Coefficients =>
        val sameMeans = this.means.equals(other.means)
        val sameVariance =
          (this.variancesOption.isDefined && other.variancesOption.isDefined &&
              this.variancesOption.get.equals(other.variancesOption.get)) ||
              (this.variancesOption.isEmpty && other.variancesOption.isEmpty)
        sameMeans && sameVariance
      case _ => false
    }
  }

  // TODO: Violation of the hashCode() contract
  override def hashCode(): Int = {
    super.hashCode()
  }
}

protected[ml] object Coefficients {
  /**
    * Create a zero coefficient vector
    *
    * @param dimension Dimensionality of the coefficient vector
    * @return Zero coefficient vector
    */
  def initializeZeroCoefficients(dimension: Int): Coefficients = {
    Coefficients(Vector.zeros[Double](dimension), variancesOption = None)
  }
}
