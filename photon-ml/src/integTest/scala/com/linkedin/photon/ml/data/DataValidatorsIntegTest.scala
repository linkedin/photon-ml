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
package com.linkedin.photon.ml.data

import org.apache.spark.rdd.RDD
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}
import com.linkedin.photon.ml.{DataValidationType, TaskType}

class DataValidatorsIntegTest extends SparkTestUtils {
  @DataProvider
  def getArgumentsForDataSanityCheck: Array[Array[Any]] = {
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
    /*
     * All RDDs have one valid point and at least one invalid point
     */
    Array(
      Array(sc.parallelize(List(lpPositiveLabel, lpBinaryLabel)),
        TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpInfLabel)),
        TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpNonBinaryLabelInfFeatures)),
        TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpInfOffset)),
        TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpBinaryLabel)),
        TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpInfLabel)),
        TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(sc.parallelize(List(lpPositiveLabel, lpNonBinaryLabelInfFeatures)),
        TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(sc.parallelize(List(lpPositiveLabel, lpInfOffset)),
        TaskType.LINEAR_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      Array(sc.parallelize(List(lpBinaryLabel)),
        TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpBinaryLabel, lpPositiveLabel)),
        TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpBinaryLabel, lpInfLabel)),
        TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpBinaryLabel, lpBinaryLabelInfFeatures)),
        TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpInfOffset)),
        TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpBinaryLabel)),
        TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, true),
      Array(sc.parallelize(List(lpBinaryLabel, lpPositiveLabel)),
        TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(sc.parallelize(List(lpBinaryLabel, lpInfLabel)),
        TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(sc.parallelize(List(lpBinaryLabel, lpBinaryLabelInfFeatures)),
        TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(sc.parallelize(List(lpBinaryLabel, lpInfOffset)),
        TaskType.LOGISTIC_REGRESSION, DataValidationType.VALIDATE_FULL, false),

      Array(sc.parallelize(List(lpPositiveLabel, lpBinaryLabel)),
        TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpInfLabel)),
        TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpNegativeLabel)),
        TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpNonBinaryLabelInfFeatures)),
        TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpInfOffset)),
        TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_DISABLED, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpBinaryLabel)),
        TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, true),
      Array(sc.parallelize(List(lpPositiveLabel, lpInfLabel)),
        TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(sc.parallelize(List(lpPositiveLabel, lpNegativeLabel)),
        TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(sc.parallelize(List(lpPositiveLabel, lpNonBinaryLabelInfFeatures)),
        TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false),
      Array(sc.parallelize(List(lpPositiveLabel, lpInfOffset)),
        TaskType.POISSON_REGRESSION, DataValidationType.VALIDATE_FULL, false)
    )
  }

  // check data provider correctness
  @Test
  def testDataProvider(): Unit = sparkTest("testDataProvider") {
    getArgumentsForDataSanityCheck
  }

  /*
  // TODO This test does not work because the spark context ends up being null in the data provider. Unlike other tests
  // we have, this test requires the spark context in the data provider and not the test. I believe this test does not
  // work because our sparkTest framework initializes the context after the data provider rather than before. I have
  // temporarily implemented the test without using DataProvider annotation
  @Test(dataProvider = "getArgumentsForDataSanityCheck")
  def testSanityCheckData(data: RDD[LabeledPoint], taskType: TaskType,
                          dataValidationType: DataValidationType, isDataSane: Boolean): Unit = sparkTest("testSanityCheckData") {
    Assert.assertEquals(DataValidators.sanityCheckData(data, taskType, dataValidationType), isDataSane)
  } */

  @Test
  def testSanityCheckData(): Unit = sparkTest("testSanityCheckData") {
    val input = getArgumentsForDataSanityCheck
    for (x <- input) {
      Assert.assertEquals(DataValidators.sanityCheckData(x(0).asInstanceOf[RDD[LabeledPoint]],
        x(1).asInstanceOf[TaskType], x(2).asInstanceOf[DataValidationType]), x(3).asInstanceOf[Boolean])
    }
  }
}
