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
import org.testng.annotations.Test

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.test.CommonTestUtils.zipWithIndex
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}

/**
 * Integration test cases for the [[AreaUnderROCCurveMultiEvaluator]].
 */
class AreaUnderROCCurveMultiEvaluatorIntegTest extends SparkTestUtils {

  var startIndex = 0

  // Normal cases
  private val labelsInNormalCase = zipWithIndex(Array[Double](1, 0, 0, 0, 1, 1, 1), startIndex)
  private val scoresInNormalCase = zipWithIndex(Array[Double](-1, -0.1, 0, 1, 2, 6, 8), startIndex)
  private val idsInNormalCase = zipWithIndex(Array.fill[String](scoresInNormalCase.length)("normal"), startIndex)
  private val expectedAUCInNormalCase = 0.75
  startIndex += labelsInNormalCase.length

  // Two identical scores with conflicting ground-truth labels
  private val labelsInCornerCase = zipWithIndex(Array[Double](0, 1, 0, 0, 1, 1, 1), startIndex)
  private val scoresInCornerCase = zipWithIndex(Array[Double](-0.1, -1, 0, -1, 2, 6, 8), startIndex)
  private val idsInCornerCase = zipWithIndex(Array.fill[String](scoresInCornerCase.length)("corner"), startIndex)
  private val expectedAUCInCornerCase = 0.791666666667
  startIndex += labelsInCornerCase.length

  // All examples have positive label
  private val positiveLabelsOnly = zipWithIndex(Array[Double](1, 1), startIndex)
  private val scoresWithPositiveLabelsOnly = zipWithIndex(Array[Double](0.5, 0.5), startIndex)
  private val idsWithPositiveLabelsOnly =
    zipWithIndex(Array.fill[String](positiveLabelsOnly.length)("all-pos"), startIndex)
  startIndex += labelsInCornerCase.length

  // All examples have negative label
  private val negativeLabelsOnly = zipWithIndex(Array[Double](0, 0), startIndex)
  private val scoresWithNegativeLabelsOnly = zipWithIndex(Array[Double](0.5, 0.5), startIndex)
  private val idsWithNegativeLabelsOnly =
    zipWithIndex(Array.fill[String](positiveLabelsOnly.length)("all-neg"), startIndex)

  /**
   * Create a new [[AreaUnderROCCurveMultiEvaluator]].
   *
   * @param labels A list of (unique sample identifier, ground truth label) pairs
   * @param ids A list of (unique sample identifier, ID) pairs
   * @return The new [[AreaUnderROCCurveMultiEvaluator]]
   */
  private def getEvaluator(labels: Seq[(UniqueSampleId, Double)], ids: Seq[(UniqueSampleId, String)]): Evaluator = {

    val defaultOffset = 0.0
    val defaultWeight = 1.0
    val labelAndOffsetAndWeights = sc.parallelize(labels).mapValues((_, defaultOffset, defaultWeight))

    new AreaUnderROCCurveMultiEvaluator(idTag = "", sc.parallelize(ids), labelAndOffsetAndWeights)
  }

  /**
   * Test that the [[AreaUnderROCCurveMultiEvaluator]] correctly computes AUC for normal and corner cases, and that it
   * correctly averages the results across IDs.
   */
  @Test
  def testEvaluateInNormalCase(): Unit = sparkTest("testEvaluateInNormalCase") {

    val labels = labelsInNormalCase ++ labelsInCornerCase
    val ids = idsInNormalCase ++ idsInCornerCase
    val scores = scoresInNormalCase ++ scoresInCornerCase
    val expectedResult = (expectedAUCInNormalCase + expectedAUCInCornerCase) / 2

    val evaluator = getEvaluator(labels, ids)
    val actualResult = evaluator.evaluate(sc.parallelize(scores.map { case (id, score) =>
      (id, score)
    }))

    assertEquals(actualResult, expectedResult, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }

  /**
   * Test that the [[AreaUnderROCCurveMultiEvaluator]] correctly filters IDs with invalid AUC results (positive only
   * case).
   */
  @Test
  def testEvaluateWithPositiveOnlyLabels(): Unit = sparkTest("testEvaluateWithPositiveOnlyLabels") {
    val labels = labelsInNormalCase ++ labelsInCornerCase ++ positiveLabelsOnly
    val ids = idsInNormalCase ++ idsInCornerCase ++ idsWithPositiveLabelsOnly
    val scores = scoresInNormalCase ++ scoresInCornerCase ++ scoresWithPositiveLabelsOnly
    val expectedResult = (expectedAUCInNormalCase + expectedAUCInCornerCase) / 2

    val evaluator = getEvaluator(labels, ids)
    val actualResult = evaluator.evaluate(sc.parallelize(scores.map { case (id, score) =>
      (id, score)
    }))

    assertEquals(actualResult, expectedResult, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }

  /**
   * Test that the [[AreaUnderROCCurveMultiEvaluator]] correctly filters IDs with invalid AUC results (negative only
   * case).
   */
  @Test
  def testEvaluateWithNegativeOnlyLabels(): Unit = sparkTest("testEvaluateWithNegativeOnlyLabels") {
    val labels = labelsInNormalCase ++ labelsInCornerCase ++ negativeLabelsOnly
    val ids = idsInNormalCase ++ idsInCornerCase ++ idsWithNegativeLabelsOnly
    val scores = scoresInNormalCase ++ scoresInCornerCase ++ scoresWithNegativeLabelsOnly
    val expectedResult = (expectedAUCInNormalCase + expectedAUCInCornerCase) / 2

    val evaluator = getEvaluator(labels, ids)
    val actualResult = evaluator.evaluate(sc.parallelize(scores.map { case (id, score) =>
      (id, score)
    }))

    assertEquals(actualResult, expectedResult, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }
}
