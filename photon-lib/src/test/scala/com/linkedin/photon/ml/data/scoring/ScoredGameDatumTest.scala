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
package com.linkedin.photon.ml.data.scoring

import breeze.linalg.DenseVector
import org.testng.Assert.{assertEquals, assertFalse, assertTrue}
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.util.MathUtils.isAlmostZero

/**
 * Unit tests for [[ScoredGameDatum]].
 */
class ScoredGameDatumTest {
  private val label = 2.3
  private val offset = 0.5
  private val weight = 1.2
  private val score = 11.0
  private val idMap = Map("src" -> "1234", "dst" -> "5678")
  private val epsilon = 1e-13

  private val datumZeroScoreOneResponse = ScoredGameDatum(1.0, 0.0, 1.0, 0.0, Map())
  private val customDatum = ScoredGameDatum(label, offset, weight, score, idMap)
  private val similarCustomDatum = ScoredGameDatum(
    label + epsilon,
    offset - epsilon,
    weight + epsilon,
    score - epsilon,
    idMap)
  private val customDatumWithNanResponse = ScoredGameDatum(Double.NaN, offset, weight, score, idMap)
  private val similarCustomDatumWithNanResponse = ScoredGameDatum(Double.NaN,
    offset + epsilon,
    weight - epsilon,
    score + epsilon,
    idMap)

  private val mapStr = idMap.toString
  private val datumZeroScoreOneResponseStr =
    s"[response=1.0, offset=0.0, weight=1.0, score=0.0, idTagToValueMap=${Map().toString}]"
  private val customDatumStr =
    s"[response=$label, offset=$offset, weight=$weight, score=$score, idTagToValueMap=$mapStr]"

  @Test
  def testEquals(): Unit = {
    assertTrue(customDatum.copy().equals(customDatum))
    assertTrue(customDatum.equals(similarCustomDatum))
    assertTrue(customDatumWithNanResponse.copy().equals(customDatumWithNanResponse))
    assertTrue(customDatumWithNanResponse.equals(similarCustomDatumWithNanResponse))

    assertFalse(customDatum.equals(datumZeroScoreOneResponse))
    assertFalse(customDatum.equals(customDatumWithNanResponse))
  }

  @Test
  def testToString(): Unit = {
    assertEquals(datumZeroScoreOneResponseStr, datumZeroScoreOneResponse.toString)
    assertEquals(customDatumStr, customDatum.toString)
  }

  @Test(dependsOnMethods = Array("testEquals"))
  def testConstructorDefaults(): Unit = {
    assertTrue(datumZeroScoreOneResponse.equals(ScoredGameDatum()))
  }

  @Test(dependsOnMethods = Array("testEquals"))
  def testApply(): Unit = {
    assertTrue(
      customDatum.equals(ScoredGameDatum(LabeledPoint(label, DenseVector[Double](0.0), offset, weight), score, idMap)))
  }
}
