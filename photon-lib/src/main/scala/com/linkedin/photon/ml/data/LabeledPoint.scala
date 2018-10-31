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

import com.linkedin.photon.ml.Types.SparkVector
import com.linkedin.photon.ml.data
import com.linkedin.photon.ml.util.{Summarizable, VectorUtils}

/**
 * Class that represents a labeled data point used for supervised learning in GAME. It has a couple fields more than
 * the spark.ml LabeledPoint (offset and weight: spark.ml has only a label and a vector of coefficients).
 *
 * @param label The label for this data point
 * @param features A vector (could be either dense or sparse) representing the features for this data point
 * @param offset The offset of this data point (e.g., is used in warm start training)
 * @param weight The weight of this data point
 */
class LabeledPoint(
  val label: Double,
  override val features: Vector[Double],
  val offset: Double = 0.0,
  override val weight: Double = 1.0) extends DataPoint(features, weight) with Summarizable {

  /**
   * A human-readable string for this labeled point.
   *
   * @return A string formatted for humans
   */
  override def toSummaryString: String = s"(label: $label offset: $offset weight: $weight features: $features)"

  /**
   * Return a string for the label and feature values in this LabeledPoint, without adornments (for easy parsing and
   * ingestion into other tools).
   *
   * The format is: label feature1Value feature2Value ...,
   * all the data separated by a space.
   *
   * @return A machine-parsable, space separated string
   */
  override def toString: String = s"$label ${features.toDenseVector.toArray.mkString(", ")}"

  /**
   * Calculate the margin, i.e. z = theta^T^x + offset.
   *
   * @param coefficients The coefficients vector
   * @return The margin
   */
  override def computeMargin(coefficients: Vector[Double]): Double = features.dot(coefficients) + offset
}

/**
 * Companion object of [[data.LabeledPoint]] for factories and pattern matching purposes
 */
object LabeledPoint {
  /**
   * Build a GAME LabeledPoint from a breeze Vector.
   *
   * @param label The label for this data point
   * @param features A vector (could be either dense or sparse) representing the features for this data point
   * @param offset The offset of this data point (e.g., is used in warm start training)
   * @param weight The weight of this data point
   * @return
   */
  def apply(label: Double, features: Vector[Double], offset: Double, weight: Double): LabeledPoint = {
    new LabeledPoint(label, features, offset, weight)
  }

  /**
   * Build a GAME LabeledPoint from a spark.ml Vector.
   *
   * @param label The label for this data point
   * @param features A vector (could be either dense or sparse) representing the features for this data point
   * @param offset The offset of this data point (e.g., is used in warm start training)
   * @param weight The weight of this data point
   * @return
   */
  def apply(
      label: Double,
      features: SparkVector,
      offset: Double = 0.0,
      weight: Double = 1.0): LabeledPoint = {
    new LabeledPoint(label, VectorUtils.mlToBreeze(features), offset, weight)
  }

  /**
   * The extractor, for pattern matching
   *
   * @return An Option containing the fields of this object
   */
  def unapply(data: LabeledPoint): Option[(Double, Vector[Double], Double, Double)] =
    Some((data.label, data.features, data.offset, data.weight))
}
