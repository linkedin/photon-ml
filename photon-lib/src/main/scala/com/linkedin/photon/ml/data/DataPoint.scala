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
package com.linkedin.photon.ml.data

import breeze.linalg.Vector

import com.linkedin.photon.ml.data

/**
 * A general data point contains features and other auxiliary information.
 *
 * @param features A vector (could be either dense or sparse) representing the features for this data point
 * @param weight The weight of this data point
 */
class DataPoint(val features: Vector[Double], val weight: Double = 1.0) extends Serializable {
  override def toString: String = {
    s"(features $features\nweight $weight)"
  }

  /**
   * Calculate the margin (i.e. z = theta^T^x).
   *
   * @param coef The coefficient vector
   * @return The margin
   */
  def computeMargin(coef: Vector[Double]): Double = features.dot(coef)
}

/**
 * Companion object of [[data.DataPoint]] for factories and pattern matching purpose.
 */
object DataPoint {
  /**
   * Apply methods give you a nice syntactic sugar for when a class or object has one main use.
   *
   * @param features Input features
   * @param weight Input weight
   * @return A new DataPoint
   */
  def apply(features: Vector[Double], weight: Double): DataPoint = {
    new DataPoint(features, weight)
  }

  /**
   * The extractor.
   *
   * @param data
   * @return The construction parameters of the given DataPoint
   */
  def unapply(data: DataPoint): Option[(Vector[Double], Double)] =
    Some((data.features, data.weight))
}
