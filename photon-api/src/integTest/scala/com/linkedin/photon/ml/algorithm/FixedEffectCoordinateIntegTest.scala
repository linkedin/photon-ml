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
package com.linkedin.photon.ml.algorithm

import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.{GameTestUtils, MathUtils}

/**
 * Integration tests for [[FixedEffectCoordinate]].
 */
class FixedEffectCoordinateIntegTest extends SparkTestUtils with GameTestUtils {

  import FixedEffectCoordinateIntegTest._

  /**
   * Test that a [[FixedEffectCoordinate]] can train a new model.
   */
  @Test(dependsOnMethods = Array("testScore"))
  def testTrainModel(): Unit = sparkTest("testUpdateModel") {

    val (coordinate, model) = generateFixedEffectCoordinateAndModel(
      FEATURE_SHARD_ID,
      NUM_TRAINING_SAMPLES,
      DIMENSIONALITY)

    // Score before model update
    val score = coordinate.score(model)
    assertTrue(score.scores.map(_._2).collect.forall(MathUtils.isAlmostZero))

    // Train models
    val (newModelWithoutInitial, _) = coordinate.trainModel()
    val (newModelWithInitial, _) = coordinate.trainModel(model)

    assertNotEquals(newModelWithoutInitial, model)
    assertNotEquals(newModelWithInitial, model)

    // Score after model update
    val newScoreWithoutInitial = coordinate.score(newModelWithoutInitial)
    val newScoreWithInitial = coordinate.score(newModelWithInitial)

    assertFalse(newScoreWithoutInitial.scores.map(_._2).collect.forall(MathUtils.isAlmostZero))
    assertFalse(newScoreWithInitial.scores.map(_._2).collect.forall(MathUtils.isAlmostZero))

    newScoreWithoutInitial
      .scores
      .join(newScoreWithInitial.scores)
      .values
      .foreach { case (score1, score2) =>
        assertEquals(score1, score2, MathConst.EPSILON)
      }
  }

  /**
   * Test that a [[FixedEffectCoordinate]] can score data using a [[com.linkedin.photon.ml.model.FixedEffectModel]].
   */
  @Test
  def testScore(): Unit = sparkTest("testScore") {

    val (coordinate, model) = generateFixedEffectCoordinateAndModel(
      FEATURE_SHARD_ID,
      NUM_TRAINING_SAMPLES,
      DIMENSIONALITY)

    val score = coordinate.score(model)
    assertEquals(score.scores.count, NUM_TRAINING_SAMPLES)
    assertTrue(score.scores.map(_._2).collect.forall(MathUtils.isAlmostZero))
  }
}

object FixedEffectCoordinateIntegTest {

  private val FEATURE_SHARD_ID = "shard1"
  private val NUM_TRAINING_SAMPLES = 1000
  private val DIMENSIONALITY = 10
}
