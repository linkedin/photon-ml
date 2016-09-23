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
package com.linkedin.photon.ml.data

import breeze.linalg.Vector
import com.linkedin.photon.ml.data


/**
 * Class that represents a labeled data point used for supervised learning
 * @param label The label for this data point
 * @param features A vector (could be either dense or sparse) representing the features for this data point
 * @param offset The offset of this data point (e.g., is used in warm start training)
 * @param weight The weight of this data point
 */
class LabeledPoint(val label: Double,
                   override val features: Vector[Double],
                   val offset: Double = 0.0,
                   override val weight: Double = 1.0) extends DataPoint(features, weight) {
  override def toString: String = {
    s"(label: $label offset: $offset weight: $weight features: $features)"
  }

  /**
   * Calculate the margin, i.e. z = theta^T^ features + offset
   * @param coef The coefficient
   * @return The margin
   */
  override def computeMargin(coef: Vector[Double]): Double = features.dot(coef) + offset
}


/**
 * Companion object of [[data.LabeledPoint]] for factories and pattern matching purpose
 */
object LabeledPoint {

  /**
   * Apply methods give you a nice syntactic sugar for when a class or object has one main use.
   */
  def apply(label: Double, features: Vector[Double], offset: Double, weight: Double): LabeledPoint = {
    new LabeledPoint(label, features, offset, weight)
  }

  /**
   * The extractor
   */
  def unapply(data: LabeledPoint): Option[(Double, Vector[Double], Double, Double)] =
    Some((data.label, data.features, data.offset, data.weight))
}
