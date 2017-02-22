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
package com.linkedin.photon.ml.supervised.regression

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * Represents a regression that predicts values for the given data set / data point using the model trained.
 */
trait Regression extends Serializable {

  /**
   * Predict values for the given data set using the model trained.
   *
   * @param features RDD representing data points' features
   * @return RDD[Double] where each entry contains the corresponding prediction
   */
  def predictAll(features: RDD[Vector[Double]]): RDD[Double] = {
    predictAllWithOffsets(features.map(feature => (feature, 0.0)))
  }

  /**
   * Predict values for the given data points with offsets of the form RDD[(feature, offset)] using the model trained.
   *
   * @param featuresWithOffsets Data points of the form RDD[(feature, offset)]
   * @return RDD[Double] where each entry contains the corresponding prediction
   */
  def predictAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double]

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features Vector representing a single data point's features
   * @return Double prediction from the trained model
   */
  def predict(features: Vector[Double]): Double = predictWithOffset(features, 0.0)

  /**
   * Predict values for a single data point with offset using the model trained.
   *
   * @param features Vector representing feature of a single data point's features
   * @param offset Offset of the data point
   * @return Double prediction from the trained model
   */
  def predictWithOffset(features: Vector[Double], offset: Double): Double
}
