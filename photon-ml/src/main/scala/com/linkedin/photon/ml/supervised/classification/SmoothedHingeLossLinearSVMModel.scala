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
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.Regression
import org.apache.spark.rdd.RDD

/**
  * Class for the classification model trained using soft hinge loss linear SVM
  *
  * @param coefficients Weights estimated for every feature
  */
class SmoothedHingeLossLinearSVMModel(override val coefficients: Vector[Double])
  extends GeneralizedLinearModel(coefficients)
  with BinaryClassifier
  with Regression
  with Serializable {

  /**
    * Compute the mean of the soft hinge loss linear SVM model
    *
    * @param features The input data point's feature
    * @param offset The input data point's offset
    * @return The mean for the passed features
    */
  override protected[ml] def computeMean(features: Vector[Double], offset: Double)
    : Double = coefficients.dot(features) + offset

  override def predictClassWithOffset(features: Vector[Double], offset: Double, threshold: Double = 0.5): Double =
    predictClass(predictWithOffset(features, offset), threshold)

  override def predictClassAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)], threshold: Double = 0.5)
    : RDD[Double] = predictAllWithOffsets(featuresWithOffsets).map(predictClass(_, threshold))

  override def predictWithOffset(features: Vector[Double], offset: Double): Double =
    computeMeanFunctionWithOffset(features, offset)

  override def predictAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double] =
    GeneralizedLinearModel.computeMeanFunctionsWithOffsets(this, featuresWithOffsets)

  private def predictClass(score: Double, threshold: Double): Double = {
    if (score < threshold) {
      BinaryClassifier.negativeClassLabel
    } else {
      BinaryClassifier.positiveClassLabel
    }
  }
}
