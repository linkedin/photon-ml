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
package com.linkedin.photon.ml.evaluation

import com.linkedin.photon.ml.test.SparkTestUtils
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst


class ShardedAreaUnderROCCurveEvaluatorTest extends SparkTestUtils {

  import TestUtilFunctions.zipWithIndex

  var startIndex = 0

  // normal cases
  val labelsInNormalCase = zipWithIndex(Array[Double](1, 0, 0, 0, 1, 1, 1), startIndex)
  val scoresInNormalCase = zipWithIndex(Array[Double](-1, -0.1, 0, 1, 2, 6, 8), startIndex)
  val idsInNormalCase = zipWithIndex(Array.fill[String](scoresInNormalCase.length)("normal"), startIndex)
  val expectedAUCInNormalCase = 0.75
  startIndex += labelsInNormalCase.length

  // if we have two identical scores with conflicting ground-truth labels
  val labelsInCornerCase = zipWithIndex(Array[Double](0, 1, 0, 0, 1, 1, 1), startIndex)
  val scoresInCornerCase = zipWithIndex(Array[Double](-0.1, -1, 0, -1, 2, 6, 8), startIndex)
  val idsInCornerCase = zipWithIndex(Array.fill[String](scoresInCornerCase.length)("corner"), startIndex)
  val expectedAUCInCornerCase = 0.79166667
  startIndex += labelsInCornerCase.length

  // where all examples have positive label
  val positiveLabelsOnly = zipWithIndex(Array[Double](1, 1), startIndex)
  val scoresWithPositiveLabelsOnly = zipWithIndex(Array[Double](0.5, 0.5), startIndex)
  val idsWithPositiveLabelsOnly = zipWithIndex(Array.fill[String](positiveLabelsOnly.length)("all-pos"), startIndex)
  startIndex += labelsInCornerCase.length

  // where all examples have negative label
  val negativeLabelsOnly = zipWithIndex(Array[Double](0, 0), startIndex)
  val scoresWithNegativeLabelsOnly = zipWithIndex(Array[Double](0.5, 0.5), startIndex)
  val idsWithNegativeLabelsOnly = zipWithIndex(Array.fill[String](positiveLabelsOnly.length)("all-neg"), startIndex)


  private def getEvaluator(labels: Seq[(Long, Double)], ids: Seq[(Long, String)]): Evaluator = {
    val defaultOffset = 0.0
    val defaultWeight = 1.0
    val labelAndOffsetAndWeights = sc.parallelize(labels).mapValues((_, defaultOffset, defaultWeight))
    new ShardedAreaUnderROCCurveEvaluator(idType = "", sc.parallelize(ids), labelAndOffsetAndWeights)
  }

  @Test
  def testEvaluateInNormalCase(): Unit = sparkTest("testEvaluateInNormalCase") {

    val labels = labelsInNormalCase ++ labelsInCornerCase
    val ids = idsInNormalCase ++ idsInCornerCase
    val scores = scoresInNormalCase ++ scoresInCornerCase
    val expectedResult = (expectedAUCInNormalCase + expectedAUCInCornerCase) / 2

    val evaluator = getEvaluator(labels, ids)
    val actualResult = evaluator.evaluate(sc.parallelize(scores))
    assertEquals(actualResult, expectedResult, MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)
  }

  @Test
  def testEvaluateWithPositiveOnlyLabels(): Unit = sparkTest("testEvaluateWitnPositiveOnlyLabels") {
    val labels = labelsInNormalCase ++ labelsInCornerCase ++ positiveLabelsOnly
    val ids = idsInNormalCase ++ idsInCornerCase ++ idsWithPositiveLabelsOnly
    val scores = scoresInNormalCase ++ scoresInCornerCase ++ scoresWithPositiveLabelsOnly

    val evaluator = getEvaluator(labels, ids)
    val actualResult = evaluator.evaluate(sc.parallelize(scores))
    assertTrue(actualResult.isNaN)
  }

  @Test
  def testEvaluateWithNegativeOnlyLabels(): Unit = sparkTest("testEvaluateWitnNegativeOnlyLabels") {
    val labels = labelsInNormalCase ++ labelsInCornerCase ++ negativeLabelsOnly
    val ids = idsInNormalCase ++ idsInCornerCase ++ idsWithNegativeLabelsOnly
    val scores = scoresInNormalCase ++ scoresInCornerCase ++ scoresWithNegativeLabelsOnly

    val evaluator = getEvaluator(labels, ids)
    val actualResult = evaluator.evaluate(sc.parallelize(scores))
    assertTrue(actualResult.isNaN)
  }
}
