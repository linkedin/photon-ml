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
package com.linkedin.photon.ml.evaluation

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Types.{REId, UniqueSampleId}
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.test.CommonTestUtils.zipWithIndex
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}

/**
 * Integration test cases for the [[AreaUnderROCCurveMultiEvaluator]].
 */
class AreaUnderROCCurveMultiEvaluatorIntegTest extends SparkTestUtils {

  private def dataHelper(
      scores: Array[Double],
      labels: Array[Double],
      id: String): (Array[(UniqueSampleId, (Double, Double, Double))], Array[(UniqueSampleId, REId)]) = {

    val data = zipWithIndex(scores
      .zip(labels)
      .map { case (score, label) =>
        (score, label, MathConst.DEFAULT_WEIGHT)
      })
    val ids = data.map { case (index, _) =>
      (index, id)
    }

    (data, ids)
  }

  /**
   * Provide test K values, IDs, scores, labels, weights, and expected results.
   */
  @DataProvider
  def evaluateInput: Array[Array[Any]] = {

    // Normal case
    val idInNormalCase = "normal"
    val scoresInNormalCase = Array[Double](-1, -0.1, 0, 1, 2, 6, 8)
    val labelsInNormalCase = Array[Double](1, 0, 0, 0, 1, 1, 1)
    val expectedAUCInNormalCase = 0.75
    val (dataInNormalCase, idsInNormalCase) = dataHelper(scoresInNormalCase, labelsInNormalCase, idInNormalCase)

    // Corner case: two identical scores with conflicting ground-truth labels
    val idInIdenticalCase = "identical"
    val scoresInIdenticalCase = Array[Double](-1, -1, -0.1, 0, 2, 6, 8)
    val labelsInIdenticalCase = Array[Double](1, 0, 0, 0, 1, 1, 1)
    val expectedAUCInIdenticalCase = 0.791666666667
    val (dataInIdenticalCase, idsInIdenticalCase) = dataHelper(
      scoresInIdenticalCase,
      labelsInIdenticalCase,
      idInIdenticalCase)

    // Corner case: all examples have positive label
    val idInAllPosCase = "all-pos"
    val scoresInAllPosCase = Array[Double](0.5, 0.5)
    val labelsInAllPosCase = Array[Double](1, 1)
    val expectedAUCInAllPosCase = 0D
    val (dataInAllPosCase, idsInAllPosCase) = dataHelper(scoresInAllPosCase, labelsInAllPosCase, idInAllPosCase)

    // Corner case: all examples have negative label
    val idInAllNegCase = "all-neg"
    val scoresInAllNegCase = Array[Double](0.5, 0.5)
    val labelsInAllNegCase = Array[Double](0, 0)
    val expectedAUCInAllNegCase = 0D
    val (dataInAllNegCase, idsInAllNegCase) = dataHelper(scoresInAllNegCase, labelsInAllNegCase, idInAllNegCase)

    Array(
      // Individual cases
      Array(idsInNormalCase, dataInNormalCase, expectedAUCInNormalCase),
      Array(idsInIdenticalCase, dataInIdenticalCase, expectedAUCInIdenticalCase),
      Array(idsInAllPosCase, dataInAllPosCase, expectedAUCInAllPosCase),
      Array(idsInAllNegCase, dataInAllNegCase, expectedAUCInAllNegCase),

      // Combined case
      Array(
        idsInNormalCase ++ idsInIdenticalCase,
        dataInNormalCase ++ dataInIdenticalCase,
        (expectedAUCInNormalCase + expectedAUCInIdenticalCase) / 2))
  }

  /**
   * Test that the [[AreaUnderROCCurveMultiEvaluator]] correctly computes AUC, and that it correctly averages the
   * results across IDs.
   *
   * @param ids Entity Ids of the records, used to group records for each unique entity
   * @param scoresAndLabelsAndWeights Predicted scores, responses, and weights for the records
   * @param expectedResult The expected precision @ k value computed for the above inputs
   */
  @Test(dataProvider = "evaluateInput")
  def testEvaluate(
      ids: Array[(UniqueSampleId, String)],
      scoresAndLabelsAndWeights: Array[(UniqueSampleId, (Double, Double, Double))],
      expectedResult: Double): Unit = sparkTest("testEvaluate") {

    val evaluator = new AreaUnderROCCurveMultiEvaluator(idTag = "", sc.parallelize(ids))
    val actualResult = evaluator.evaluate(sc.parallelize(scoresAndLabelsAndWeights))

    assertEquals(actualResult, expectedResult, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }
}
