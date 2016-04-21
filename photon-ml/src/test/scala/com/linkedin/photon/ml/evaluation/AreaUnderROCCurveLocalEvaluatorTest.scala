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

import scala.collection.Map

import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst


/**
 * Test functions in [[AreaUnderROCCurveLocalEvaluator]]
 */
class AreaUnderROCCurveLocalEvaluatorTest {

  private def arrayToIndexedMap(array: Array[Double]): Map[Long, Double] = {
    array.zipWithIndex.map { case (label, index) => (index.toLong, label) }.toMap
  }

  @Test
  def testEvaluate(): Unit = {

    // normal cases
    val labelsInNormalCase = arrayToIndexedMap(Array[Double](0, 1, 0, 0, 1, 1, 1))
    val scoresInNormalCase = arrayToIndexedMap(Array[Double](-0.1, -1, 0, 1, 2, 6, 8))
    val groundTruthFromRInNormalCase = 0.75
    val computedAUCInNormalCase = new AreaUnderROCCurveLocalEvaluator(labelsInNormalCase)
        .evaluate(scoresInNormalCase)
    assertEquals(groundTruthFromRInNormalCase, computedAUCInNormalCase, MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)

    // where all examples have positive label
    val positiveLabelsOnly = arrayToIndexedMap(Array[Double](1, 1))
    val scoresWithPositiveLabelsOnly = arrayToIndexedMap(Array[Double](0.5, 0.5))
    val computedAUCWithPositiveLabelsOnly = new AreaUnderROCCurveLocalEvaluator(positiveLabelsOnly)
        .evaluate(scoresWithPositiveLabelsOnly)
    assertTrue(computedAUCWithPositiveLabelsOnly.isNaN)

    // where all examples have negative label
    val negativeLabelsOnly = arrayToIndexedMap(Array[Double](0, 0))
    val scoresWithNegativeLabelsOnly = arrayToIndexedMap(Array[Double](0.5, 0.5))
    val computedAUCWithNegativeLabelsOnly = new AreaUnderROCCurveLocalEvaluator(negativeLabelsOnly)
        .evaluate(scoresWithNegativeLabelsOnly)
    assertTrue(computedAUCWithNegativeLabelsOnly.isNaN)
  }
}
