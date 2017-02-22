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

import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.test.CommonTestUtils.getScoreLabelAndWeights

/**
 * Test functions in [[AreaUnderROCCurveLocalEvaluator]].
 */
class AreaUnderROCCurveLocalEvaluatorTest {

  @Test
  def testEvaluate(): Unit = {

    // normal cases
    val labelsInNormalCase = Array[Double](1, 0, 0, 0, 1, 1, 1)
    val scoresInNormalCase = Array[Double](-1, -0.1, 0, 1, 2, 6, 8)
    val scoreLabelAndWeightsInNormalCase = getScoreLabelAndWeights(scoresInNormalCase, labelsInNormalCase)
    val expectedAUCInNormalCase = 0.75
    val computedAUCInNormalCase = AreaUnderROCCurveLocalEvaluator.evaluate(scoreLabelAndWeightsInNormalCase)
    assertEquals(computedAUCInNormalCase, expectedAUCInNormalCase, MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)

    // if we have two identical scores with conflicting ground-truth labels
    val labelsInCornerCase1 = Array[Double](0, 1, 0, 0, 1, 1, 1)
    val scoresInCornerCase1 = Array[Double](-0.1, -1, 0, -1, 2, 6, 8)
    val scoreLabelAndWeightsInCornerCase1 = getScoreLabelAndWeights(scoresInCornerCase1, labelsInCornerCase1)
    val expectedAUCInCornerCase1 = 0.79166667
    val computedAUCInCornerCase1 = AreaUnderROCCurveLocalEvaluator.evaluate(scoreLabelAndWeightsInCornerCase1)
    assertEquals(computedAUCInCornerCase1, expectedAUCInCornerCase1, MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)

    // where all examples have positive label
    val positiveLabelsOnly = Array[Double](1, 1)
    val scoresWithPositiveLabelsOnly = Array[Double](0.5, 0.5)
    val scoreLabelAndWeightsPositiveLabelsOnly =
      getScoreLabelAndWeights(scoresWithPositiveLabelsOnly, positiveLabelsOnly)
    val computedAUCWithPositiveLabelsOnly =
      AreaUnderROCCurveLocalEvaluator.evaluate(scoreLabelAndWeightsPositiveLabelsOnly)
    assertTrue(computedAUCWithPositiveLabelsOnly.isNaN)

    // where all examples have negative label
    val negativeLabelsOnly = Array[Double](0, 0)
    val scoresWithNegativeLabelsOnly = Array[Double](0.5, 0.5)
    val scoreLabelAndWeightsNegativeLabelsOnly =
      getScoreLabelAndWeights(scoresWithNegativeLabelsOnly, negativeLabelsOnly)
    val computedAUCWithNegativeLabelsOnly =
      AreaUnderROCCurveLocalEvaluator.evaluate(scoreLabelAndWeightsNegativeLabelsOnly)
    assertTrue(computedAUCWithNegativeLabelsOnly.isNaN)
  }
}
