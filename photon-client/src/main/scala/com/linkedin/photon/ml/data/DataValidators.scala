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

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.util.Logging
import com.linkedin.photon.ml.{DataValidationType, TaskType}

/**
 * A collection of methods used to validate data before applying ML algorithms.
 */
object DataValidators extends Logging {

  // (Validator, Error Message) pairs
  val baseValidators: List[((LabeledPoint => Boolean), String)] = List(
    (finiteFeatures, "Data contains row(s) with non-finite feature(s)"),
    (finiteOffset, "Data contains row(s) with non-finite offset(s)"),
    (finiteWeight, "Data contains row(s) with non-finite weight(s)"))
  val linearRegressionValidators: List[((LabeledPoint => Boolean), String)] =
    (finiteLabel _, "Data contains row(s) with non-finite label(s)") :: baseValidators
  val logisticRegressionValidators: List[((LabeledPoint => Boolean), String)] =
    (binaryLabel _, "Data contains row(s) with non-binary label(s)") :: baseValidators
  val poissonRegressionValidators: List[((LabeledPoint => Boolean), String)] =
    (finiteLabel _, "Data contains row(s) with non-finite label(s)") ::
      (nonNegativeLabels _, "Data contains row(s) with negative label(s)") ::
      baseValidators

  /**
   * Verify that a labeled data point has a finite label.
   *
   * @param labeledPoint The input data point
   * @return Whether the label of the input data point is finite
   */
  def finiteLabel(labeledPoint: LabeledPoint): Boolean = !(labeledPoint.label.isNaN || labeledPoint.label.isInfinite)

  /**
   * Verify that a labeled data point has a binary label.
   *
   * @param labeledPoint The input data point
   * @return Whether the label of the input data point is binary
   */
  def binaryLabel(labeledPoint: LabeledPoint): Boolean =
    BinaryClassifier.positiveClassLabel == labeledPoint.label || BinaryClassifier.negativeClassLabel == labeledPoint.label

  /**
   * Verify that a labeled data point has a non-negative label.
   *
   * @param labeledPoint The input data point
   * @return Whether the label of the input data point is non-negative
   */
  def nonNegativeLabels(labeledPoint: LabeledPoint): Boolean = labeledPoint.label >= 0

  /**
   * Verify that the feature values of a data point are finite.
   *
   * @param labeledPoint The input data point
   * @return Whether all feature values for the input data point are finite
   */
  def finiteFeatures(labeledPoint: LabeledPoint): Boolean =
    labeledPoint.features.iterator.forall { case (_, value) =>
      !(value.isNaN || value.isInfinite)
    }

  /**
   * Verify that a data point has a finite offset.
   *
   * @param labeledPoint The input data point
   * @return Whether the offset of the input data point is finite
   */
  def finiteOffset(labeledPoint: LabeledPoint): Boolean = !(labeledPoint.offset.isNaN || labeledPoint.offset.isInfinite)

  /**
   * Verify that a data point has a finite weight.
   *
   * @param labeledPoint The input data point
   * @return Whether the weight of the input data point is finite
   */
  def finiteWeight(labeledPoint: LabeledPoint): Boolean = !(labeledPoint.weight.isNaN || labeledPoint.weight.isInfinite)

  /**
   * Validate a data set using one or more data point validators.
   *
   * @param dataSet The input data set
   * @param perSampleValidators A list of (data validator, error message) pairs
   * @return The list of validation error messages for the input data
   */
  private def validateData(
      dataSet: RDD[LabeledPoint],
      perSampleValidators: List[((LabeledPoint => Boolean), String)]): Seq[String] =
    perSampleValidators
      .map { case (validator, msg) =>
        val validatorBroadcast = dataSet.sparkContext.broadcast(validator)
        val result = dataSet.aggregate(true)(
          seqOp = (result, dataPoint) => result && validatorBroadcast.value(dataPoint),
          combOp = (result1, result2) => result1 && result2)

        (result, msg)
      }
      .filterNot(_._1)
      .map(_._2)

  /**
   * Validate a full or sampled data set using the set of data point validators relevant to the training problem.
   *
   * @param inputData The input data set
   * @param taskType The training task type
   * @param dataValidationType The validation intensity
   * @throws IllegalArgumentException if one or more of the data validations failed
   */
  def sanityCheckData(
      inputData: RDD[LabeledPoint],
      taskType: TaskType,
      dataValidationType: DataValidationType): Unit = {

    val validators: List[((LabeledPoint => Boolean), String)] = taskType match {
      case TaskType.LINEAR_REGRESSION => linearRegressionValidators
      case TaskType.LOGISTIC_REGRESSION => logisticRegressionValidators
      case TaskType.POISSON_REGRESSION => poissonRegressionValidators
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => logisticRegressionValidators
    }

    // Check the data properties
    val dataErrors = dataValidationType match {
      case DataValidationType.VALIDATE_FULL =>
        validateData(inputData, validators)

      case DataValidationType.VALIDATE_SAMPLE =>
        validateData(inputData.sample(withReplacement = false, fraction = 0.10), validators)

      case DataValidationType.VALIDATE_DISABLED =>
        Seq()
    }

    if (dataErrors.nonEmpty) {
      throw new IllegalArgumentException(s"Data Validation failed:\n${dataErrors.mkString("\n")}")
    }
  }
}
