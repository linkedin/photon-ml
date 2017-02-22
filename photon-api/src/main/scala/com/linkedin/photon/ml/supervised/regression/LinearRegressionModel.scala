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
package com.linkedin.photon.ml.supervised.regression

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.TaskType._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Class for the classification model trained using Logistic Regression.
 *
 * @param coefficients Weights estimated for every feature
 */
class LinearRegressionModel(override val coefficients: Coefficients)
  extends GeneralizedLinearModel(coefficients)
  with Regression
  with Serializable {

  /**
   *
   * @return The model type
   */
  override def modelType: TaskType = LINEAR_REGRESSION

  /**
   * Compute the mean of the linear regression model.
   *
   * @param features The input data point's features
   * @param offset The input data point's offset
   * @return The mean for the passed features
   */
  override protected[ml] def computeMean(features: Vector[Double], offset: Double)
    : Double = coefficients.computeScore(features) + offset

  /**
   *
   * @param updatedCoefficients
   * @return A new generalized linear model with the passed coefficients
   */
  override def updateCoefficients(updatedCoefficients: Coefficients): LinearRegressionModel =
    new LinearRegressionModel(updatedCoefficients)

  /**
   *
   * @param other Some object
   * @return Whether this object can equal the other object
   */
  override def canEqual(other: Any): Boolean = other.isInstanceOf[LinearRegressionModel]

  /**
   *
   * @return A summary of the object in string representation
   */
  override def toSummaryString: String =
    s"Linear Regression Model with the following coefficients:\n${coefficients.toSummaryString}"

  /**
   *
   * @param features vector representing feature of a single data point's features
   * @param offset offset of the data point
   * @return Double prediction from the trained model
   */
  override def predictWithOffset(features: Vector[Double], offset: Double): Double =
    computeMeanFunctionWithOffset(features, offset)

  /**
   *
   * @param featuresWithOffsets data points of the form RDD[(feature, offset)]
   * @return RDD[Double] where each entry contains the corresponding prediction
   */
  override def predictAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double] =
    GeneralizedLinearModel.computeMeanFunctionsWithOffsets(this, featuresWithOffsets)
}

object LinearRegressionModel {

  /**
   * Create a new linear regression model with the provided coefficients (means) and variances.
   *
   * @param coefficients The feature coefficient means and variances for the model
   * @return A linear regression model
   */
  def apply(coefficients: Coefficients): GeneralizedLinearModel = new LinearRegressionModel(coefficients)
}
