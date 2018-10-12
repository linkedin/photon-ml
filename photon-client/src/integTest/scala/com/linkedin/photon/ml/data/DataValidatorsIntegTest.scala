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

import scala.util.{Failure, Success, Try}

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._
import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}
import com.linkedin.photon.ml.util.VectorUtils
import com.linkedin.photon.ml.{DataValidationType, TaskType}

/**
 * Integration tests for [[DataValidators]].
 */
class DataValidatorsIntegTest extends SparkTestUtils {

  import DataValidatorsIntegTest._

  /**
   * Helper function to generate [[org.apache.spark.sql.DataFrame]] rows.
   *
   * @param features1 First sample features vector
   * @param features2 Second sample features vector
   * @param response Sample response
   * @param offset Sample offset
   * @param weight Sample weight
   * @return Input data grouped into a [[Row]] with the expected schema
   */
  private def getRow(
      features1: Vector,
      features2: Vector,
      response: Double = 1D,
      offset: Double = 0D,
      weight: Double = 1D): Row =
    new GenericRowWithSchema(Array[Any](response, offset, weight, features1, features2), ROW_SCHEMA)

  /**
   * Generate input data for [[DataValidators.sanityCheckData]].
   */
  @DataProvider
  def sanityCheckDataInput: Array[Array[Any]] = {

    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 1, 20)
    val validVector = vectors.head
    val invalidVector = vectors.last

    // LabeledPoints with valid vectors
    val positiveLabel = new LabeledPoint(5D, validVector)
    val negativeLabel = new LabeledPoint(-5D, validVector)
    val binaryLabel = new LabeledPoint(BinaryClassifier.positiveClassLabel, validVector)
    val zeroLabel = new LabeledPoint(0D, validVector)
    val nanLabel = new LabeledPoint(Double.NaN, validVector)

    // LabeledPoint with invalid vectors
    val badFeatures = new LabeledPoint(BinaryClassifier.positiveClassLabel, invalidVector)

    // LabeledPoint with invalid offset
    val badOffset = new LabeledPoint(BinaryClassifier.positiveClassLabel, validVector, Double.NaN)

    // LabeledPoint with invalid weight
    val badWeight = new LabeledPoint(BinaryClassifier.positiveClassLabel, validVector, 0D, Double.NaN)

    Array(
      // Test linear regression checks for finite labels
      Array(
        Seq(positiveLabel, negativeLabel, binaryLabel, zeroLabel),
        TaskType.LINEAR_REGRESSION,
        DataValidationType.VALIDATE_FULL,
        true),
      Array(Seq(nanLabel), TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test logistic regression checks for binary label
      Array(Seq(binaryLabel, zeroLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, true),
      Array(Seq(positiveLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(negativeLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(nanLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test smoothed hinge loss checks for binary label
      Array(
        Seq(binaryLabel, zeroLabel),
        TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM,
        DataValidationType.VALIDATE_FULL,
        true),
      Array(Seq(positiveLabel), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(negativeLabel), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(nanLabel), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),

      // Test Poisson regression checks for non-negative label
      Array(
        Seq(positiveLabel, binaryLabel, zeroLabel),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL,
        true),
      Array(Seq(negativeLabel), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(nanLabel), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test all task types require finite features
      Array(Seq(badFeatures), TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeatures), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeatures), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeatures), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test all task types require finite offset
      Array(Seq(badOffset), TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badOffset), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badOffset), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badOffset), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test all task types require valid weight
      Array(Seq(badWeight), TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badWeight), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badWeight), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badWeight), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test that even one bad sample causes failure
      Array(
        Seq(positiveLabel, binaryLabel, zeroLabel),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL,
        true),
      Array(
        Seq(positiveLabel, binaryLabel, zeroLabel, negativeLabel),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL,
        false),

      // Test that VALIDATE_DISABLED will ignore bad rows
      Array(Seq(positiveLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(positiveLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_DISABLED, true)
    )
  }

  /**
   * Generate input data for [[DataValidators.sanityCheckDataFrameForTraining]].
   */
  @DataProvider
  def sanityCheckDataFrameForTrainingInput: Array[Array[Any]] = {

    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 1, 20)
    val validVector = VectorUtils.breezeToMl(vectors.head)
    val invalidVector = VectorUtils.breezeToMl(vectors.last)

    // Rows with valid vectors
    val positiveLabel = getRow(validVector, validVector, 5D)
    val negativeLabel = getRow(validVector, validVector, -5D)
    val binaryLabel = getRow(validVector, validVector, BinaryClassifier.positiveClassLabel)
    val zeroLabel = getRow(validVector, validVector, 0D)
    val nanLabel = getRow(validVector, validVector, Double.NaN)

    // Rows with invalid vectors
    val badFeatures1 = getRow(invalidVector, validVector, BinaryClassifier.positiveClassLabel)
    val badFeatures2 = getRow(validVector, invalidVector, BinaryClassifier.positiveClassLabel)
    val badFeaturesBoth = getRow(invalidVector, invalidVector, BinaryClassifier.positiveClassLabel)

    // Row with invalid offset
    val badOffset = getRow(validVector, validVector, offset = Double.NaN)

    // Row with invalid weight
    val badWeight = getRow(validVector, validVector, weight = Double.NaN)

    Array(
      // Test linear regression checks for finite labels
      Array(
        Seq(positiveLabel, negativeLabel, binaryLabel, zeroLabel),
        TaskType.LINEAR_REGRESSION,
        DataValidationType.VALIDATE_FULL,
        true),
      Array(Seq(nanLabel), TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test logistic regression checks for binary label
      Array(Seq(binaryLabel, zeroLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, true),
      Array(Seq(positiveLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(negativeLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(nanLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test smoothed hinge loss checks for binary label
      Array(
        Seq(binaryLabel, zeroLabel),
        TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM,
        DataValidationType.VALIDATE_FULL,
        true),
      Array(Seq(positiveLabel), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(negativeLabel), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(nanLabel), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),

      // Test Poisson regression checks for non-negative label
      Array(
        Seq(positiveLabel, binaryLabel, zeroLabel),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL,
        true),
      Array(Seq(negativeLabel), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(nanLabel), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test all task types require finite features
      Array(Seq(badFeatures1), TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeatures2), TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeaturesBoth), TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeatures1), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeatures2), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeaturesBoth), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeatures1), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeatures2), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeaturesBoth), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeatures1), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeatures2), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badFeaturesBoth), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test all task types require finite offset
      Array(Seq(badOffset), TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badOffset), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badOffset), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badOffset), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test all task types require valid weight
      Array(Seq(badWeight), TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badWeight), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badWeight), TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(badWeight), TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      // Test that even one bad sample causes failure
      Array(
        Seq(positiveLabel, binaryLabel, zeroLabel),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL,
        true),
      Array(
        Seq(positiveLabel, binaryLabel, zeroLabel, negativeLabel),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL,
        false),

      // Test that VALIDATE_DISABLED will ignore bad rows
      Array(Seq(positiveLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(Seq(positiveLabel), TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_DISABLED, true)
    )
  }

  /**
   * Test that well-formed datasets will not cause errors during validation and poorly-formed datasets will.
   *
   * @param input Input data
   * @param taskType Training task
   * @param validationType Type of validation to perform
   * @param expectedResult Whether or not validation errors are expected for the input data
   */
  @Test(dataProvider = "sanityCheckDataInput")
  def testSanityCheckData(
      input: Seq[LabeledPoint],
      taskType: TaskType,
      validationType: DataValidationType,
      expectedResult: Boolean): Unit = sparkTest("testSanityCheckData") {

    val rdd = sc.parallelize(input)

    val actualResult = Try(DataValidators.sanityCheckData(rdd, taskType, validationType)) match {
      case Success(_) => true
      case Failure(_: IllegalArgumentException) => false
      case Failure(e) => throw new MatchError(s"Unexpected Error: ${e.getMessage}")
    }

    assertEquals(actualResult, expectedResult)
  }

  /**
   * Test that well-formed datasets will not cause errors during validation and poorly-formed datasets will.
   *
   * @param input Input data
   * @param taskType Training task
   * @param validationType Type of validation to perform
   * @param expectedResult Whether or not validation errors are expected for the input data
   */
  @Test(dataProvider = "sanityCheckDataFrameForTrainingInput")
  def testSanityCheckDataFrameForTraining(
      input: Seq[Row],
      taskType: TaskType,
      validationType: DataValidationType,
      expectedResult: Boolean): Unit = sparkTest("testSanityCheckDataFrameForTraining") {

    val dataFrame = sparkSession.createDataFrame(sc.parallelize(input), ROW_SCHEMA)

    val tryResult = Try(
      DataValidators.sanityCheckDataFrameForTraining(
        dataFrame,
        taskType,
        validationType,
        INPUT_COLUMN_NAMES,
        FEATURE_COLUMNS))
    val actualResult = tryResult match {
      case Success(_) => true
      case Failure(_: IllegalArgumentException) => false
      case Failure(e) => throw new MatchError(s"Unexpected Error: ${e.getMessage}")
    }

    assertEquals(actualResult, expectedResult)
  }
}

object DataValidatorsIntegTest {

  private val INPUT_COLUMN_NAMES = InputColumnsNames()

  private val RESPONSE = InputColumnsNames.RESPONSE.toString
  private val OFFSET = InputColumnsNames.OFFSET.toString
  private val WEIGHT = InputColumnsNames.WEIGHT.toString
  private val FEATURES_BAG_1 = "featureBag1"
  private val FEATURES_BAG_2 = "featureBag2"
  private val FEATURE_COLUMNS = Set(FEATURES_BAG_1, FEATURES_BAG_2)

  private val RESPONSE_COLUMN = StructField(RESPONSE, DoubleType)
  private val OFFSET_COLUMN = StructField(OFFSET, DoubleType)
  private val WEIGHT_COLUMN = StructField(WEIGHT, DoubleType)
  private val FEATURES_COLUMN_1 = StructField(FEATURES_BAG_1, VectorType)
  private val FEATURES_COLUMN_2 = StructField(FEATURES_BAG_2, VectorType)

  private val ROW_SCHEMA =
    new StructType(Array(RESPONSE_COLUMN, OFFSET_COLUMN, WEIGHT_COLUMN, FEATURES_COLUMN_1, FEATURES_COLUMN_2))
}