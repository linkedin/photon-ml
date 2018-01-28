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

import com.linkedin.photon.ml.test.CommonTestUtils.zipWithIndex
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}

/**
 * Integration test cases for the [[PrecisionAtKMultiEvaluator]].
 */
class PrecisionAtKMultiEvaluatorIntegTest extends SparkTestUtils {

  /**
   * Provide test K values, labels, IDs, scores, and expected results.
   */
  @DataProvider
  def getEvaluateTestCases: Array[Array[Any]] = {
    // A trivial case but with different Ks
    val trivialScores = zipWithIndex(Array(1.0, 0.5, 0.0))
    val trivialIds = zipWithIndex(Array("1", "1", "1"))
    val labelsWithPosOn1 = zipWithIndex(Array(1.0, 0.0, 0.0))
    val labelsWithPosOn2 = zipWithIndex(Array(0.0, 1.0, 0.0))
    val labelsWithPosOn3 = zipWithIndex(Array(0.0, 0.0, 1.0))
    val labelsWithNoPos = zipWithIndex(Array(0.0, 0.0, 0.0))
    val ks = (1 to 4).toArray
    val trivialCase1 = ks.map{ k =>
      val expectedResult = 1.0 / k
      Array(k, labelsWithPosOn1, trivialIds, trivialScores, expectedResult)
    }
    val trivialCase2 = ks.map{ k =>
      val expectedResult = if (k == 1) 0.0 else 1.0 / k
      Array(k, labelsWithPosOn2, trivialIds, trivialScores, expectedResult)
    }
    val trivialCase3 = ks.map{ k =>
      val expectedResult = if (k <= 2) 0.0 else 1.0 / k
      Array(k, labelsWithPosOn3, trivialIds, trivialScores, expectedResult)
    }
    val trivialCase4 = ks.map{ k =>
      val expectedResult = 0.0
      Array(k, labelsWithNoPos, trivialIds, trivialScores, expectedResult)
    }

    // A case adopted from Metronome
    val labels = zipWithIndex(Array[Double](0, 0, 1, 0, 1, 0, 0, 1, 1, 0))
    val ids = zipWithIndex(Array("1", "1", "1", "1", "2", "2", "2", "2", "2", "2"))
    val scores = zipWithIndex(Array(1, 0.75, 0.5, 0.25, 1, 0.75, 0.5, 0.25, 0.2, 0.1))
    val expectedResults = Array(0.5, 0.25, 1.0/3, 0.375, 0.4, 1.0/3)
    val metronomeCase = (1 to 6).toArray.map { k =>
      Array(k, labels, ids, scores, expectedResults(k-1))
    }

    trivialCase1 ++ trivialCase2 ++ trivialCase3 ++ trivialCase4 ++ metronomeCase
  }

  /**
   * Test that the [[PrecisionAtKMultiEvaluator]] correctly computes precision @ K for trivial and non-trivial cases,
   * and that it correctly averages the results across IDs.
   *
   * @param k Value at which to compute precision
   * @param labels Responses for records, assumed to be sorted by decreasing score
   * @param ids Entity Ids of the records, used to group records for each unique entity
   * @param scores Predicted scores for the records
   * @param expectedResult The expected precision @ k value computed for the above inputs
   */
  @Test(dataProvider = "getEvaluateTestCases")
  def testEvaluate(
      k: Int,
      labels: Array[(Long, Double)],
      ids: Array[(Long, String)],
      scores: Array[(Long, Double)],
      expectedResult: Double): Unit = sparkTest("testEvaluate") {

    val defaultOffset = 0.0
    val defaultWeight = 1.0
    val labelAndOffsetAndWeights = sc.parallelize(labels).mapValues((_, defaultOffset, defaultWeight))
    val evaluator = new PrecisionAtKMultiEvaluator(k, idTag = "", sc.parallelize(ids), labelAndOffsetAndWeights)
    val actualResult = evaluator.evaluate(sc.parallelize(scores.map { case (id, score) =>
      (id, score)
    }))

    assertEquals(actualResult, expectedResult, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }
}
