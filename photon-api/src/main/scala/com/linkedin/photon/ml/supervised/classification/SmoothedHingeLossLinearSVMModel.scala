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
package com.linkedin.photon.ml.supervised.classification

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.TaskType._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.Regression

/**
 * Class for the classification model trained using soft hinge loss linear SVM.
 *
 * @param coefficients Weights estimated for every feature
 */
class SmoothedHingeLossLinearSVMModel(override val coefficients: Coefficients)
  extends GeneralizedLinearModel(coefficients)
  with BinaryClassifier
  with Regression
  with Serializable {

  /**
   * Check the model type.
   *
   * @return The model type
   */
  override def modelType: TaskType = SMOOTHED_HINGE_LOSS_LINEAR_SVM

  /**
   * Compute the mean response of the smoothed hinge loss linear SVM model.
   *
   * @param features The input data point's feature
   * @param offset The input data point's offset
   * @return The mean for the passed features
   */
  override protected[ml] def computeMean(features: Vector[Double], offset: Double): Double =
    coefficients.computeScore(features) + offset

  /**
   * Create a new model of the same type with updated coefficients.
   *
   * @param updatedCoefficients The new coefficients
   * @return A new generalized linear model with the passed coefficients
   */
  override def updateCoefficients(updatedCoefficients: Coefficients): SmoothedHingeLossLinearSVMModel =
    new SmoothedHingeLossLinearSVMModel(updatedCoefficients)

  /**
   * Compares two [[SmoothedHingeLossLinearSVMModel]] objects.
   *
   * @param other Some other object
   * @return True if the both models conform to the equality contract and have the same model coefficients, false
   *         otherwise
   */
  override def equals(other: Any): Boolean = other match {
    case that: SmoothedHingeLossLinearSVMModel => super.equals(that)
    case _ => false
  }

  /**
   * Build a human-readable summary for the object.
   *
   * @return A summary of the object in string representation
   */
  override def toSummaryString: String =
    s"Smoothed Hinge Loss Linear SVM Model with the following coefficients:\n${coefficients.toSummaryString}"

  /**
   * Predict values for a single data point with offset.
   *
   * @param features Vector a single data point's features
   * @param offset Offset of the data point
   * @param threshold Threshold that separates positive predictions from negative predictions. An example with
   *                  prediction score greater than or equal to this threshold is identified as positive, and negative
   *                  otherwise.
   * @return Predicted category from the trained model
   */
  override def predictClassWithOffset(features: Vector[Double], offset: Double, threshold: Double = 0.5): Double =
    classifyScore(predictWithOffset(features, offset), threshold)

  /**
   * Predict values for the given data points with offsets of the form RDD[(feature, offset)].
   *
   * @param featuresWithOffsets Data points of the form RDD[(feature, offset)]
   * @param threshold Threshold that separates positive predictions from negative predictions. An example with
   *                  prediction score greater than or equal to this threshold is identified as positive, and negative
   *                  otherwise.
   * @return An RDD[Double] where each entry contains the corresponding prediction
   */
  override def predictClassAllWithOffsets(
      featuresWithOffsets: RDD[(Vector[Double], Double)],
      threshold: Double = 0.5): RDD[Double] =
    predictAllWithOffsets(featuresWithOffsets).map(classifyScore(_, threshold))

  /**
   * Predict values for a single data point with offset.
   *
   * @param features Vector representing feature of a single data point's features
   * @param offset Offset of the data point
   * @return Double prediction from the trained model
   */
  override def predictWithOffset(features: Vector[Double], offset: Double): Double =
    computeMeanFunctionWithOffset(features, offset)

  /**
   * Predict values for the given data points with offsets of the form RDD[(feature, offset)].
   *
   * @param featuresWithOffsets Data points of the form RDD[(feature, offset)]
   * @return RDD[Double] where each entry contains the corresponding prediction
   */
  override def predictAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double] =
    GeneralizedLinearModel.computeMeanFunctionsWithOffsets(this, featuresWithOffsets)

  /**
   * Predict the binary class of a score, given a threshold.
   *
   * @param score The score
   * @param threshold The threshold for a positive class label
   * @return A positive label if the score is greater than or equal to the threshold, a negative label otherwise
   */
  private def classifyScore(score: Double, threshold: Double): Double = {
    if (score < threshold) {
      BinaryClassifier.negativeClassLabel
    } else {
      BinaryClassifier.positiveClassLabel
    }
  }
}

object SmoothedHingeLossLinearSVMModel {

  /**
   * Create a new smoothed hinge loss SVM model with the provided coefficients (means) and variances.
   *
   * @param coefficients The feature coefficient means and variances for the model
   * @return A smoothed hinge loss SVM model
   */
  def apply(coefficients: Coefficients): SmoothedHingeLossLinearSVMModel =
    new SmoothedHingeLossLinearSVMModel(coefficients)
}
