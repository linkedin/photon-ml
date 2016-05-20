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
package com.linkedin.photon.ml.supervised.model

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * GeneralizedLinearModel (GLM) represents a model trained using GeneralizedLinearAlgorithm.
 * Reference: [[http://en.wikipedia.org/wiki/Generalized_linear_model]].
 * Note that this class is modified based on MLLib's GeneralizedLinearModel.
 * @param coefficients The generalized linear model's coefficients (or called weights in some scenarios) of the features
 */
abstract class GeneralizedLinearModel(val coefficients: Vector[Double]) extends Serializable {

  protected[ml] def computeMean(coefficients: Vector[Double], features: Vector[Double], offset: Double): Double

  /**
   * Compute the value of the mean function of the generalized linear model given one data point using the estimated
   * coefficients and intercept
   * @param features vector representing a single data point's features
   * @return Computed mean function value
   */
  def computeMeanFunction(features: Vector[Double]): Double = computeMeanFunctionWithOffset(features, 0.0)

  /**
   * Compute the value of the mean function of the generalized linear model given one data point using the estimated
   * coefficients and intercept
   * @param features vector representing a single data point's features
   * @param offset offset of the data point
   * @return Computed mean function value
   */
  def computeMeanFunctionWithOffset(features: Vector[Double], offset: Double): Double =
    computeMean(coefficients, features, offset)

  /**
   * Compute the value of the mean functions of the generalized linear model given a RDD of data points using the
   * estimated coefficients and intercept
   * @param features RDD representing data points' features
   * @return Computed mean function value
   */
  def computeMeanFunctions(features: RDD[Vector[Double]]): RDD[Double] =
    computeMeanFunctionsWithOffsets(features.map(feature => (feature, 0.0)))

  /**
   * Compute the value of the mean functions of the generalized linear model given a RDD of data points using the
   * estimated coefficients and intercept
   * @param featuresWithOffsets Data points of the form RDD[(feature, offset)]
   * @return Computed mean function value
   */
  def computeMeanFunctionsWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double] = {
    val broadcastedCoefficients = featuresWithOffsets.context.broadcast(coefficients)

    featuresWithOffsets.map {
      case (features, offset) => computeMean(broadcastedCoefficients.value, features, offset)
    }
  }

  /**
   * Validate coefficients and offset. Child classes should add additional checks.
   */
  def validateCoefficients(): Unit = {
    val msg : StringBuilder = new StringBuilder()
    var valid : Boolean = true

    coefficients.foreachPair( (idx, value) => {
      if (!java.lang.Double.isFinite(value)) {
        valid = false
        msg.append("Index [" + idx + "] has value [" + value + "]\n")
      }
    })

    if (!valid) {
      throw new IllegalStateException("Detected invalid coefficients / offset: " + msg.toString())
    }
  }

  /**
   * Use String interpolation over format. It's a bit more concise and is checked at compile time (e.g. forgetting an
   * argument would be a compile error).
   */
  override def toString: String = {
    s"coefficients: $coefficients"
  }
}
