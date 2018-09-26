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

import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.test.CommonTestUtils.zipWithIndex
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}

/**
 * Integration tests for [[PrecisionAtKMultiEvaluator]].
 */
class PrecisionAtKMultiEvaluatorIntegTest extends SparkTestUtils {

  /**
   * Provide test K values, IDs, scores, labels, weights, and expected results.
   */
  @DataProvider
  def evaluateInput: Array[Array[Any]] = {

    val trivialIds = Array("1", "1", "1")
    val trivialScores = Array(1.0, 0.5, 0.0)

    val labelsWithPosOn1 = Array(1.0, 0.0, 0.0)
    val labelsWithPosOn2 = Array(0.0, 1.0, 0.0)
    val labelsWithPosOn3 = Array(0.0, 0.0, 1.0)
    val labelsWithNoPos = Array(0.0, 0.0, 0.0)

    val kValues = (1 to 4).toArray

    // Trivial cases with different K values
    val trivialCases = kValues
      .flatMap { k =>
        val expectedResultNoPos = 0.0
        val expectedResultPos1 = 1.0 / k
        val expectedResultPos2 = if (k == 1) 0.0 else 1.0 / k
        val expectedResultPos3 = if (k <= 2) 0.0 else 1.0 / k

        Seq(
          (k, labelsWithNoPos, expectedResultNoPos),
          (k, labelsWithPosOn1, expectedResultPos1),
          (k, labelsWithPosOn2, expectedResultPos2),
          (k, labelsWithPosOn3, expectedResultPos3))
      }
      .map { case (k, trivialLabels, expectedResult) =>
        val data = trivialLabels
          .zipWithIndex
          .map { case (label, index) =>
            (index.toLong, (trivialScores(index), label, MathConst.DEFAULT_WEIGHT))
          }

        Array(k, zipWithIndex(trivialIds), data, expectedResult)
      }

    val ids = Array("1", "1", "1", "1", "2", "2", "2", "2", "2", "2")
    val labels = Array[Double](0, 0, 1, 0, 1, 0, 0, 1, 1, 0)
    val scores = Array(1, 0.75, 0.5, 0.25, 1, 0.75, 0.5, 0.25, 0.2, 0.1)
    val data = zipWithIndex(scores.zip(labels).map(pair => (pair._1, pair._2, MathConst.DEFAULT_WEIGHT)))
    val expectedResults = Array(0.5, 0.25, 1D/3, 0.375, 0.4, 1D/3)
    val complexCase = (1 to 6).toArray.map { k =>
      Array(k, zipWithIndex(ids), data, expectedResults(k - 1))
    }

    trivialCases ++ complexCase
  }

  /**
   * Test that the [[PrecisionAtKMultiEvaluator]] correctly computes precision @ K for trivial and non-trivial cases,
   * and that it correctly averages the results across IDs.
   *
   * @param k Value at which to compute precision
   * @param ids Entity Ids of the records, used to group records for each unique entity
   * @param scoresAndLabelsAndWeights Predicted scores, responses, and weights for the records
   * @param expectedResult The expected precision @ k value computed for the above inputs
   */
  @Test(dataProvider = "evaluateInput")
  def testEvaluate(
      k: Int,
      ids: Array[(UniqueSampleId, String)],
      scoresAndLabelsAndWeights: Array[(UniqueSampleId, (Double, Double, Double))],
      expectedResult: Double): Unit = sparkTest("testEvaluate") {

    val evaluator = new PrecisionAtKMultiEvaluator(k, idTag = "", sc.parallelize(ids))
    val actualResult = evaluator.evaluate(sc.parallelize(scoresAndLabelsAndWeights))

    assertEquals(actualResult, expectedResult, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }
}
