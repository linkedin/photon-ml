/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

/**
 * Class for the classification model trained using Logistic Regression
 * @param coefficients Weights estimated for every feature
 * @param intercept Intercept computed for this model (Option)
 * @author xazhang
 */
class LinearRegressionModel(override val coefficients: Vector[Double], override val intercept: Option[Double])
  extends GeneralizedLinearModel(coefficients, intercept) with Regression with Serializable {

  override def predictWithOffset(features: Vector[Double], offset: Double): Double = {
    computeMeanFunctionWithOffset(features, offset)
  }

  override def predictAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double] = {
    computeMeanFunctionsWithOffsets(featuresWithOffsets)
  }

  /**
   * Compute the mean of the linear regression model
   * @param coefficients the estimated features' coefficients
   * @param intercept the estimated model intercept
   * @param features the input data point's feature
   * @param offset the input data point's offset
   * @return
   */
  override protected def computeMean(coefficients: Vector[Double], intercept: Option[Double], features: Vector[Double], offset: Double): Double = {
    coefficients.dot(features) + intercept.getOrElse(0.0) + offset
  }
}