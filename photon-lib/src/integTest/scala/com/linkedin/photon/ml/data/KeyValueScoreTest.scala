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

import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * Simple tests for [[KeyValueScore]].
 */
class KeyValueScoreTest extends SparkTestUtils {

  /**
   * Generate a [[KeyValueScore]] with the given scores.
   *
   * @param scores The given scores
   * @return The [[KeyValueScore]] generated with the given scores
   */
  private def generateKeyValueScore(scores: Array[Double]): KeyValueScore = {
    new KeyValueScore(sc.parallelize(scores.zipWithIndex.map { case (score, uniqueId) =>
      (uniqueId.toLong, ScoredGameDatum(score = score)) }))
  }

  /**
   * Generate a [[KeyValueScore]] with the given keys and values.
   *
   * @param keys The given keys
   * @param values The given values
   * @return The [[KeyValueScore]] generated with the given keys and values
   */
  private def generateKeyValueScore(keys: Array[Long], values: Array[Double]): KeyValueScore =
    new KeyValueScore(sc.parallelize(keys.zip(values.map(v => ScoredGameDatum(score = v)))))

  @Test
  def testEquals(): Unit = sparkTest("testEqualsForKeyValueScores") {
    //case 1: key value scores of length 0
    val emptyScores1 = generateKeyValueScore(scores = Array[Double]())
    val emptyScores2 = generateKeyValueScore(scores = Array[Double]())
    assertEquals(emptyScores1, emptyScores2)

    //case 2: key value scores with different length
    val nonEmptyScores1 = generateKeyValueScore(scores = Array[Double](1, 2))
    val nonEmptyScores2 = generateKeyValueScore(scores = Array[Double](1, 2, 3))
    assertNotEquals(nonEmptyScores1, nonEmptyScores2)

    //case 3: key value scores with same keys but different values
    val sharedKeys = Array[Long](1, 3, 5)
    val values1 = Array[Double](0, 0, 0)
    val values2 = Array[Double](0, 1, 0)
    val keyValueScore1 = generateKeyValueScore(sharedKeys, values1)
    val keyValueScore2 = generateKeyValueScore(sharedKeys, values2)
    assertNotEquals(keyValueScore1, keyValueScore2)

    //case 4: key value scores with different keys but same values
    val keys1 = Array[Long](0, 1, 2)
    val keys2 = Array[Long](1, 2, 3)
    val sharedValues = Array[Double](1, 2, 3)
    val keyValueScore3 = generateKeyValueScore(keys1, sharedValues)
    val keyValueScore4 = generateKeyValueScore(keys2, sharedValues)
    assertNotEquals(keyValueScore3, keyValueScore4)

    //case 5: same keys and same values, but the order of key/value pairs are different
    val keys3 = Array[Long](0, 1, 2)
    val keys4 = Array[Long](2, 1, 0)
    val values3 = Array[Double](4, 5, 6)
    val values4 = Array[Double](6, 5, 4)
    val keyValueScore5 = generateKeyValueScore(keys3, values3)
    val keyValueScore6 = generateKeyValueScore(keys4, values4)
    assertEquals(keyValueScore5, keyValueScore6)
  }

  @Test
  def testPlus(): Unit = sparkTest("testPlusForKeyValueScores") {
    //case 1: both key value scores are of length 0
    val emptyScores = generateKeyValueScore(scores = Array[Double]())
    assertEquals(emptyScores - emptyScores, emptyScores)

    //case 2: one of the scores are empty
    val nonEmptyScores = generateKeyValueScore(scores = Array[Double](1, 2))
    assertEquals(emptyScores + nonEmptyScores, nonEmptyScores)

    //case 3: when both scores are non-empty
    val keys1 = Array[Long](0, 1, 2)
    val keys2 = Array[Long](2, 3, 4)
    val values1 = Array[Double](1, -1, 1)
    val values2 = Array[Double](-1, 0, 1)
    val keyValueScore1 = generateKeyValueScore(keys1, values1)
    val keyValueScore2 = generateKeyValueScore(keys2, values2)
    val expectedKeys = Array[Long](0, 1, 2, 3, 4)
    val expectedValues = Array[Double](1, -1, 0, 0, 1)
    val expectedKeyValueScore = generateKeyValueScore(expectedKeys, expectedValues)
    assertEquals(keyValueScore1 + keyValueScore2, expectedKeyValueScore)
  }

  @Test
  def testMinus(): Unit = sparkTest("testMinusForKeyValueScores") {
    //case 1: both key value scores are of length 0
    val emptyScores = generateKeyValueScore(scores = Array[Double]())
    assertEquals(emptyScores - emptyScores, emptyScores)

    //case 2: one of the scores are empty
    val nonEmptyScores = generateKeyValueScore(scores = Array[Double](1, 2))
    assertEquals(nonEmptyScores - emptyScores, nonEmptyScores)
    val expectedNonEmptyScores = generateKeyValueScore(scores = Array[Double](-1, -2))
    assertEquals(emptyScores - nonEmptyScores, expectedNonEmptyScores)

    //case 3: when both scores are non-empty
    val keys1 = Array[Long](0, 1, 2)
    val keys2 = Array[Long](2, 3, 4)
    val values1 = Array[Double](1, -1, 1)
    val values2 = Array[Double](-1, 0, 1)
    val keyValueScore1 = generateKeyValueScore(keys1, values1)
    val keyValueScore2 = generateKeyValueScore(keys2, values2)
    val expectedKeys1 = Array[Long](0, 1, 2, 3, 4)
    val expectedValues1 = Array[Double](1, -1, 2, 0, -1)
    val expectedKeyValueScore1 = generateKeyValueScore(expectedKeys1, expectedValues1)
    assertEquals(keyValueScore1 - keyValueScore2, expectedKeyValueScore1)
    val expectedKeys2 = Array[Long](0, 1, 2, 3, 4)
    val expectedValues2 = Array[Double](-1, 1, -2, 0, 1)
    val expectedKeyValueScore2 = generateKeyValueScore(expectedKeys2, expectedValues2)
    assertEquals(keyValueScore2 - keyValueScore1, expectedKeyValueScore2)
  }
}
