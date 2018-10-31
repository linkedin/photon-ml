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

import scala.util.{Success, Try}

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vectors
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}
import com.linkedin.photon.ml.{DataValidationType, TaskType}
import org.apache.spark.sql.types._

/**
 * Integration tests for [[DataValidators]].
 */
class DataValidatorsIntegTest extends SparkTestUtils {

  @DataProvider
  def getSuccessArgumentsForSanityCheckData: Array[Array[Any]] = {
    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 1, 20)
    val validVector = vectors.head
    val invalidVector = vectors.last

    // labeled points with valid vectors
    val lpPositiveLabel = new LabeledPoint(5.0, validVector)
    val lpNegativeLabel = new LabeledPoint(-5.0, validVector)
    val lpBinaryLabel = new LabeledPoint(BinaryClassifier.negativeClassLabel, validVector)

    // labeled point with invalid label
    val lpInfLabel = new LabeledPoint(Double.PositiveInfinity, validVector)

    // labeled point with invalid offset
    val lpInfOffset = new LabeledPoint(BinaryClassifier.positiveClassLabel, validVector, Double.NaN)

    // labeled points with invalid vectors
    val lpNonBinaryLabelInfFeatures = new LabeledPoint(-2.0, invalidVector)
    val lpBinaryLabelInfFeatures = new LabeledPoint(BinaryClassifier.negativeClassLabel, invalidVector)

    Assert.assertNotNull(sc)

    // All RDDs have one valid point and at least one invalid point
    Array(
      Array(
        sc.parallelize(List(lpPositiveLabel, lpBinaryLabel)),
        TaskType.LINEAR_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpInfLabel)),
        TaskType.LINEAR_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpNonBinaryLabelInfFeatures)),
        TaskType.LINEAR_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpInfOffset)),
        TaskType.LINEAR_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpBinaryLabel)),
        TaskType.LINEAR_REGRESSION,
        DataValidationType.VALIDATE_FULL),

      Array(
        sc.parallelize(List(lpBinaryLabel)),
        TaskType.LOGISTIC_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpBinaryLabel, lpPositiveLabel)),
        TaskType.LOGISTIC_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpBinaryLabel, lpInfLabel)),
        TaskType.LOGISTIC_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpBinaryLabel, lpBinaryLabelInfFeatures)),
        TaskType.LOGISTIC_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpInfOffset)),
        TaskType.LOGISTIC_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpBinaryLabel)),
        TaskType.LOGISTIC_REGRESSION,
        DataValidationType.VALIDATE_FULL),

      Array(
        sc.parallelize(List(lpPositiveLabel, lpBinaryLabel)),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpInfLabel)),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpNegativeLabel)),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpNonBinaryLabelInfFeatures)),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpInfOffset)),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_DISABLED),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpBinaryLabel)),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL)
    )
  }

  @DataProvider
  def getFailureArgumentsForSanityCheckData: Array[Array[Any]] = {
    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 1, 20)
    val validVector = vectors.head
    val invalidVector = vectors.last

    // labeled points with valid vectors
    val lpPositiveLabel = new LabeledPoint(5.0, validVector)
    val lpNegativeLabel = new LabeledPoint(-5.0, validVector)
    val lpBinaryLabel = new LabeledPoint(BinaryClassifier.negativeClassLabel, validVector)

    // labeled point with invalid label
    val lpInfLabel = new LabeledPoint(Double.PositiveInfinity, validVector)

    // labeled point with invalid offset
    val lpInfOffset = new LabeledPoint(BinaryClassifier.positiveClassLabel, validVector, Double.NaN)

    // labeled points with invalid vectors
    val lpNonBinaryLabelInfFeatures = new LabeledPoint(-2.0, invalidVector)
    val lpBinaryLabelInfFeatures = new LabeledPoint(BinaryClassifier.negativeClassLabel, invalidVector)

    Assert.assertNotNull(sc)

    // All RDDs have one valid point and at least one invalid point
    Array(
      Array(
        sc.parallelize(List(lpPositiveLabel, lpInfLabel)),
        TaskType.LINEAR_REGRESSION,
        DataValidationType.VALIDATE_FULL, false),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpNonBinaryLabelInfFeatures)),
        TaskType.LINEAR_REGRESSION,
        DataValidationType.VALIDATE_FULL, false),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpInfOffset)),
        TaskType.LINEAR_REGRESSION,
        DataValidationType.VALIDATE_FULL, false),

      Array(
        sc.parallelize(List(lpBinaryLabel, lpPositiveLabel)),
        TaskType.LOGISTIC_REGRESSION,
        DataValidationType.VALIDATE_FULL, false),
      Array(
        sc.parallelize(List(lpBinaryLabel, lpInfLabel)),
        TaskType.LOGISTIC_REGRESSION,
        DataValidationType.VALIDATE_FULL, false),
      Array(
        sc.parallelize(List(lpBinaryLabel, lpBinaryLabelInfFeatures)),
        TaskType.LOGISTIC_REGRESSION,
        DataValidationType.VALIDATE_FULL, false),
      Array(
        sc.parallelize(List(lpBinaryLabel, lpInfOffset)),
        TaskType.LOGISTIC_REGRESSION,
        DataValidationType.VALIDATE_FULL, false),

      Array(
        sc.parallelize(List(lpPositiveLabel, lpInfLabel)),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL, false),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpNegativeLabel)),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL, false),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpNonBinaryLabelInfFeatures)),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL, false),
      Array(
        sc.parallelize(List(lpPositiveLabel, lpInfOffset)),
        TaskType.POISSON_REGRESSION,
        DataValidationType.VALIDATE_FULL, false)
    )
  }

  // TODO: Unlike other tests, these tests require the Spark context in the data provider. These tests do not work as
  // usual because our sparkTest framework initializes the context after the data provider rather than before. The tests
  // have been temporarily implemented without using the DataProvider annotation.

  @Test
  def testSuccessSanityCheckData(): Unit = sparkTest("testSanityCheckData") {
    val input = getSuccessArgumentsForSanityCheckData

    for (x <- input) {
      DataValidators.sanityCheckData(
        x(0).asInstanceOf[RDD[LabeledPoint]],
        x(1).asInstanceOf[TaskType],
        x(2).asInstanceOf[DataValidationType])
    }
  }

  @Test
  def testFailureSanityCheckData(): Unit = sparkTest("testSanityCheckData") {
    val input = getFailureArgumentsForSanityCheckData

    for (x <- input) {
      val result = Try(
        DataValidators.sanityCheckData(
          x(0).asInstanceOf[RDD[LabeledPoint]],
          x(1).asInstanceOf[TaskType],
          x(2).asInstanceOf[DataValidationType]))

      result match {
        case Success(_) => throw new IllegalArgumentException("Unexpected success for bad data")
        case _ =>
      }
    }
  }

  @Test
  def testSuccessSanityCheckDataFrame(): Unit = sparkTest("testSanityCheckDataFrame") {

    val schema = new StructType(Array(StructField(InputColumnsNames.RESPONSE.toString, DoubleType),
      StructField(InputColumnsNames.WEIGHT.toString, DoubleType),
      StructField(InputColumnsNames.OFFSET.toString, DoubleType),
      StructField(InputColumnsNames.FEATURES_DEFAULT.toString, VectorType)))

    val input = getSuccessArgumentsForSanityCheckData

    for (x <- input) {
      val labeledPoints = x(0).asInstanceOf[RDD[LabeledPoint]]

      val rows = labeledPoints.map(lp => {
        val features = Vectors.dense(lp.features.toArray).toSparse
        Row(lp.label, lp.weight, lp.offset, features)
      })

      val inputColumnsNames = InputColumnsNames()
      val featureNames = Set(InputColumnsNames.FEATURES_DEFAULT.toString)

      DataValidators.sanityCheckDataFrameForTraining(
        sparkSession.createDataFrame(rows, schema),
        x(1).asInstanceOf[TaskType],
        x(2).asInstanceOf[DataValidationType],
        inputColumnsNames,
        featureNames)
    }
  }

  @Test
  def testFailureSanityCheckDataFrame(): Unit = sparkTest("testSanityCheckDataFrame") {

    val schema = new StructType(Array(StructField(InputColumnsNames.RESPONSE.toString, DoubleType),
      StructField(InputColumnsNames.WEIGHT.toString, DoubleType),
      StructField(InputColumnsNames.OFFSET.toString, DoubleType),
      StructField(InputColumnsNames.FEATURES_DEFAULT.toString, VectorType)))

    val input = getFailureArgumentsForSanityCheckData

    for (x <- input) {
      val labeledPoints = x(0).asInstanceOf[RDD[LabeledPoint]]

      val rows = labeledPoints.map(lp => {
        val features = Vectors.dense(lp.features.toArray).toSparse
        Row(lp.label, lp.weight, lp.offset, features)
      })

      val inputColumnsNames = InputColumnsNames()
      val featureNames = Set(InputColumnsNames.FEATURES_DEFAULT.toString)

      val result = Try(DataValidators.sanityCheckDataFrameForTraining(
        sparkSession.createDataFrame(rows, schema),
        x(1).asInstanceOf[TaskType],
        x(2).asInstanceOf[DataValidationType],
        inputColumnsNames,
        featureNames))

      result match {
        case Success(_) => throw new IllegalArgumentException("Unexpected success for bad data")
        case _ =>
      }
    }
  }
}