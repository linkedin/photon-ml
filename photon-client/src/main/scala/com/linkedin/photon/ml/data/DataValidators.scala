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
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.mllib.linalg.SparseVector

import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types.FeatureShardId
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.util.Logging
import com.linkedin.photon.ml.{DataValidationType, TaskType}

import com.linkedin.photon.ml.constants.MathConst

/**
 * A collection of methods used to validate data before applying ML algorithms.
 */
object DataValidators extends Logging {

  // (Validator, Error Message) pairs
  val baseValidators: List[((LabeledPoint => Boolean), String)] = List(
    (finiteFeatures, "Data contains row(s) with invalid (+/- Inf or NaN) feature(s)"),
    (finiteOffset, "Data contains row(s) with invalid (+/- Inf or NaN) offset(s)"),
    (nonNegativeWeight, "Data contains row(s) with invalid (-, Inf, or NaN) weight(s)"))
  val linearRegressionValidators: List[((LabeledPoint => Boolean), String)] =
    (finiteLabel _, "Data contains row(s) with invalid (+/- Inf or NaN) label(s)") :: baseValidators
  val logisticRegressionValidators: List[((LabeledPoint => Boolean), String)] =
    (binaryLabel _, "Data contains row(s) with non-binary label(s)") :: baseValidators
  val poissonRegressionValidators: List[((LabeledPoint => Boolean), String)] =
    (nonNegativeLabel _, "Data contains row(s) with invalid (-, Inf, or NaN) label(s)") :: baseValidators

  // (Validator, Input Column Name, Error Message) triples
  val dataFrameBaseValidators: List[(((Row, String) => Boolean), InputColumnsNames.Value, String)] = List(
    (rowHasFiniteFeatures,
      InputColumnsNames.FEATURES_DEFAULT,
      "Data contains row(s) with invalid (+/- Inf or NaN) feature(s)"),
    (rowHasFiniteColumn, InputColumnsNames.OFFSET, "Data contains row(s) with invalid (+/- Inf or NaN) offset(s)"),
    (rowHasNonNegativeWeight, InputColumnsNames.WEIGHT, "Data contains row(s) with invalid (-, Inf, or NaN) weight(s)"))
  val dataFrameLinearRegressionValidators: List[(((Row, String) => Boolean), InputColumnsNames.Value, String)] =
    (rowHasFiniteColumn _, InputColumnsNames.RESPONSE, "Data contains row(s) with invalid (+/- Inf or NaN) label(s)") ::
      dataFrameBaseValidators
  val dataFrameLogisticRegressionValidators: List[(((Row, String) => Boolean), InputColumnsNames.Value, String)] =
    (rowHasBinaryLabel _, InputColumnsNames.RESPONSE, "Data contains row(s) with non-binary label(s)") ::
      dataFrameBaseValidators
  val dataFramePoissonRegressionValidators: List[(((Row, String) => Boolean), InputColumnsNames.Value, String)] =
    (rowHasNonNegativeLabel _, InputColumnsNames.RESPONSE, "Data contains row(s) with invalid (-, Inf, or Nan) label(s)") ::
      dataFrameBaseValidators

  /**
   * Verify that a row has a finite column.
   *
   * @param row The input row from a data frame
   * @param inputColumnName The column name we want to validate
   * @return Whether the column of the row is finite
   */
  def rowHasFiniteColumn(row: Row, inputColumnName: String): Boolean =
    row.getAs[Any](inputColumnName) match {
      case col: Double => !(col.isNaN || col.isInfinite)
      case _ => false
    }

  /**
   * Verify that a labeled data point has a finite label.
   *
   * @param labeledPoint The input data point
   * @return Whether the label of the input data point is finite
   */
  def finiteLabel(labeledPoint: LabeledPoint): Boolean =
    !(labeledPoint.label.isNaN || labeledPoint.label.isInfinite)

  /**
   * Verify that a labeled data point has a binary label.
   *
   * @param labeledPoint The input data point
   * @return Whether the label of the input data point is binary
   */
  def binaryLabel(labeledPoint: LabeledPoint): Boolean =
    (BinaryClassifier.positiveClassLabel == labeledPoint.label) ||
      (BinaryClassifier.negativeClassLabel == labeledPoint.label)

  /**
   * Verify that a row has a binary label.
   *
   * @param row The input row from a data frame
   * @param inputColumnName The column name we want to validate
   * @return Whether the label column of the row is binary
   */
  def rowHasBinaryLabel(row: Row, inputColumnName: String): Boolean =
    row.getAs[Any](inputColumnName) match {
      case label: Double => BinaryClassifier.positiveClassLabel == label || BinaryClassifier.negativeClassLabel == label
      case _ => false
    }

  /**
   * Verify that a labeled data point has a non-negative label.
   *
   * @param labeledPoint The input data point
   * @return Whether the label of the input data point is non-negative
   */
  def nonNegativeLabel(labeledPoint: LabeledPoint): Boolean =
    finiteLabel(labeledPoint) && (labeledPoint.label >= 0)

  /**
   * Verify that a row has a non-negative label.
   *
   * @param row The input row from a data frame
   * @param inputColumnName The column name we want to validate
   * @return Whether the label column of the row is non-negative
   */
  def rowHasNonNegativeLabel(row: Row, inputColumnName: String): Boolean =
    row.getAs[Any](inputColumnName) match {
      case label: Double => !(label.isNaN || label.isInfinite) && (label >= 0)
      case _ => false
    }

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
   * Verify that the feature values of a row are finite.
   *
   * @param row The input row from a data frame
   * @param inputColumnName The column name we want to validate
   * @return Whether all feature values of the row are finite
   */
  def rowHasFiniteFeatures(row: Row, inputColumnName: String): Boolean =
    row.getAs[Any](inputColumnName) match {
      case features: SparseVector => features.values.forall(value => !(value.isNaN || value.isInfinite))
      case _ => false
    }

  /**
   * Verify that a data point has a finite offset.
   *
   * @param labeledPoint The input data point
   * @return Whether the offset of the input data point is finite
   */
  def finiteOffset(labeledPoint: LabeledPoint): Boolean =
    !(labeledPoint.offset.isNaN || labeledPoint.offset.isInfinite)

  /**
   * Verify that a data point has a finite weight.
   *
   * @param labeledPoint The input data point
   * @return Whether the weight of the input data point is finite
   */
  def nonNegativeWeight(labeledPoint: LabeledPoint): Boolean =
    !(labeledPoint.weight.isNaN || labeledPoint.weight.isInfinite) && (labeledPoint.weight > MathConst.EPSILON)

  /**
   * Verify that a row has a positive weight.
   *
   * @param row The input row from a data frame
   * @param inputColumnName The column name we want to validate
   * @return Whether the weight column of the row is positive
   */
  def rowHasNonNegativeWeight(row: Row, inputColumnName: String): Boolean =
    row.getAs[Any](inputColumnName) match {
      // Weight should be significantly larger than 0
      case weight: Double =>  !(weight.isNaN || weight.isInfinite) && (weight > MathConst.EPSILON)
      case _ => false
    }

  /**
   * Validate a dataset using one or more data point validators.
   *
   * @param dataset The input dataset
   * @param perSampleValidators A list of (data validator, error message) pairs
   * @return The list of validation error messages for the input data
   */
  private def validateData(
      dataset: RDD[LabeledPoint],
      perSampleValidators: List[((LabeledPoint => Boolean), String)]): Seq[String] =
    perSampleValidators
      .map { case (validator, msg) =>
        val validatorBroadcast = dataset.sparkContext.broadcast(validator)
        val result = dataset.aggregate(true)(
          seqOp = (result, dataPoint) => result && validatorBroadcast.value(dataPoint),
          combOp = (result1, result2) => result1 && result2)

        (result, msg)
      }
      .filterNot(_._1)
      .map(_._2)

  /**
   * Validate a full or sampled dataset using the set of data point validators relevant to the training problem.
   *
   * @param inputData The input dataset
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

  /**
   * Validate a data frame using one or more data point validators.
   *
   * @param dataset The input data frame
   * @param perSampleValidators A list of (data validator, input column name, error message) triples
   * @param inputColumnsNames Column names for the provided data frame
   * @param featureSectionKeys Column names for the feature columns in the provided data frame
   * @return The list of validation error messages for the input data frame
   */
  private def validateDataFrame(
      dataset: DataFrame,
      perSampleValidators: List[(((Row, String) => Boolean), InputColumnsNames.Value, String)],
      inputColumnsNames: InputColumnsNames,
      featureSectionKeys: Set[FeatureShardId]): Seq[String] = {

    val columns = dataset.columns
    dataset
      .rdd
      .flatMap { r =>
        perSampleValidators
          .flatMap { case (validator, columnName, msg) =>
            if (columnName == InputColumnsNames.FEATURES_DEFAULT) {
              featureSectionKeys.map(features => (validator(r, features), msg))
            } else {
              val result = if (columns.contains(inputColumnsNames(columnName))) {
                (validator(r, inputColumnsNames(columnName)), msg)
              } else {
                (true, "")
              }

              Seq(result)
            }
          }
          .filterNot(_._1)
          .map(_._2)
      }
      .collect()
  }

  /**
   * Validate a full or sampled data frame using the set of data point validators relevant to the training problem.
   *
   * @param inputData The input data frame
   * @param taskType The training task type
   * @param dataValidationType The validation intensity
   * @param inputColumnsNames Column names for the provided data frame
   * @param featureSectionKeys Column names for the feature columns in the provided data frame
   * @throws IllegalArgumentException if one or more of the data validations failed
   */
  def sanityCheckDataFrameForTraining(
      inputData: DataFrame,
      taskType: TaskType,
      dataValidationType: DataValidationType,
      inputColumnsNames: InputColumnsNames,
      featureSectionKeys: Set[FeatureShardId]): Unit = {

    val validators: List[(((Row, String) => Boolean), InputColumnsNames.Value, String)] = taskType match {
      case TaskType.LINEAR_REGRESSION => dataFrameLinearRegressionValidators
      case TaskType.LOGISTIC_REGRESSION => dataFrameLogisticRegressionValidators
      case TaskType.POISSON_REGRESSION => dataFramePoissonRegressionValidators
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => dataFrameLogisticRegressionValidators
    }

    // Check the data properties
    val dataErrors = dataValidationType match {
      case DataValidationType.VALIDATE_FULL =>
        validateDataFrame(
          inputData,
          validators,
          inputColumnsNames,
          featureSectionKeys)

      case DataValidationType.VALIDATE_SAMPLE =>
        validateDataFrame(
          inputData.sample(withReplacement = false, fraction = 0.10),
          validators,
          inputColumnsNames,
          featureSectionKeys)

      case DataValidationType.VALIDATE_DISABLED =>
        Seq()
    }

    if (dataErrors.nonEmpty) {
      throw new IllegalArgumentException(s"Data Validation failed:\n${dataErrors.mkString("\n")}")
    }
  }

  /**
   * Validate a full or sampled data frame using the set of data point validators relevant to the scoring problem.
   *
   * @param inputData The input data frame
   * @param dataValidationType The validation intensity
   * @param inputColumnsNames Column names for the provided data frame
   * @param featureSectionKeys Column names for the feature columns in the provided data frame
   * @param taskTypeOpt The training task type (no task type represents no response data)
   * @throws IllegalArgumentException if one or more of the data validations failed
   */
  def sanityCheckDataFrameForScoring(
      inputData: DataFrame,
      dataValidationType: DataValidationType,
      inputColumnsNames: InputColumnsNames,
      featureSectionKeys: Set[FeatureShardId],
      taskTypeOpt: Option[TaskType] = None): Unit = taskTypeOpt match {
    case Some(taskType) =>
      sanityCheckDataFrameForTraining(inputData, taskType, dataValidationType, inputColumnsNames, featureSectionKeys)

    case None =>
      // Check the data properties
      val dataErrors = dataValidationType match {
        case DataValidationType.VALIDATE_FULL =>
          validateDataFrame(
            inputData,
            dataFrameBaseValidators,
            inputColumnsNames,
            featureSectionKeys)

        case DataValidationType.VALIDATE_SAMPLE =>
          validateDataFrame(
            inputData.sample(withReplacement = false, fraction = 0.10),
            dataFrameBaseValidators,
            inputColumnsNames,
            featureSectionKeys)

        case DataValidationType.VALIDATE_DISABLED =>
          Seq()
      }

      if (dataErrors.nonEmpty) {
        throw new IllegalArgumentException(s"Data Validation failed:\n${dataErrors.mkString("\n")}")
      }
  }
}
