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
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.{GameTestUtils, MathUtils}

/**
 * Integration tests for the [[RandomEffectModelCoordinate]] implementation.
 */
class RandomEffectModelCoordinateIntegTest extends SparkTestUtils with GameTestUtils {

  import RandomEffectModelCoordinateIntegTest._

  @DataProvider
  def numEntitiesDataProvider(): Array[Array[Any]] = Array(Array(1), Array(2), Array(10))

  /**
   * Test that a [[RandomEffectModelCoordinate]] can score data using a [[com.linkedin.photon.ml.model.RandomEffectModel]].
   *
   * @param numEntities The number of unique per-entity models to train
   */
  @Test(dataProvider = "numEntitiesDataProvider")
  def testScore(numEntities: Int): Unit = sparkTest("testScore") {

    val (coordinate, model) = generateRandomEffectCoordinateAndModel(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_ID,
      numEntities,
      NUM_TRAINING_SAMPLES,
      DIMENSIONALITY)

    val score = coordinate.score(model)

    assertEquals(score.scoresRdd.count, numEntities * NUM_TRAINING_SAMPLES)
    assertTrue(score.scoresRdd.map(_._2).collect.forall(MathUtils.isAlmostZero))
  }
}

object RandomEffectModelCoordinateIntegTest {

  private val RANDOM_EFFECT_TYPE = "random-effect-1"
  private val FEATURE_SHARD_ID = "shard1"
  private val NUM_TRAINING_SAMPLES = 1000
  private val DIMENSIONALITY = 10
}
