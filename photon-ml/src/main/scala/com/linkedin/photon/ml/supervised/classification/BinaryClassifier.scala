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
package com.linkedin.photon.ml.supervised.classification

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * Represents a binary classifier, with 1 representing the positive label and 0 representing the negative label.
 * @author xazhang
 */
trait BinaryClassifier extends Serializable {

  /**
   * Predict values for a single data point using the model trained.
   * @param features vector representing a single data point's features
   * @param threshold threshold that separates positive predictions from negative predictions. An example with prediction
   *                  score greater than or equal to this threshold is identified as positive, and negative otherwise.
   * @return predicted category from the trained model
   */
  def predictClass(features: Vector[Double], threshold: Double): Double = predictClassWithOffset(features, 0.0, threshold)

  /**
   * Predict values for a single data point with offset using the model trained.
   * @param features vector representing feature of a single data point's features
   * @param offset offset of the data point
   * @param threshold threshold that separates positive predictions from negative predictions. An example with prediction
   *                  score greater than or equal to this threshold is identified as positive, and negative otherwise.
   * @return predicted category from the trained model
   */
  def predictClassWithOffset(features: Vector[Double], offset: Double, threshold: Double): Double

  /**
   * Predict values for the given data points of the form RDD[feature] using the model trained.
   * @param features RDD representing data points' features
   * @param threshold threshold that separates positive predictions from negative predictions. An example with prediction
   *                  score greater than or equal to this threshold is identified as positive, and negative otherwise.
   * @return an RDD[Double] where each entry contains the corresponding prediction
   */
  def predictClassAllWithThreshold(features: RDD[Vector[Double]], threshold: Double): RDD[Double] = {
    predictClassAllWithOffsets(features.map(feature => (feature, 0.0)), threshold)
  }

  /**
   * Predict values for the given data points with offsets of the form RDD[(feature, offset)] using the model trained.
   * @param featuresWithOffsets data points of the form RDD[(feature, offset)]
   * @param threshold threshold that separates positive predictions from negative predictions. An example with prediction
   *                  score greater than or equal to this threshold is identified as positive, and negative otherwise.
   * @return an RDD[Double] where each entry contains the corresponding prediction
   */
  def predictClassAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)], threshold: Double): RDD[Double]
}

object BinaryClassifier {
  val positiveClassLabel = 1.0
  val negativeClassLabel = 0.0
}