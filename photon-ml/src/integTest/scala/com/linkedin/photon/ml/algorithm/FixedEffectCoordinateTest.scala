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
package com.linkedin.photon.ml.algorithm

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.GameTestUtils
import com.linkedin.photon.ml.test.SparkTestUtils

/**
  * Tests for the FixedEffectCoordinate implementation
  */
class FixedEffectCoordinateTest extends SparkTestUtils with GameTestUtils {
  import FixedEffectCoordinateTest._

  val featureShardId = "shard1"

  @Test
  def testUpdateModel(): Unit = sparkTest("testUpdateModel") {
    val (coordinate, model) = generateFixedEffectCoordinateAndModel(
      featureShardId, NumTrainingSamples, Dimensionality)

    // Score before model update
    val score = coordinate.score(model)
    assertTrue(score.scores.map(_._2).collect.forall(_ == 0.0))

    // Update model
    val (newModel, _) = coordinate.updateModel(model)
    assertNotEquals(newModel, model)

    // Score after model update
    val newScore = coordinate.score(newModel)
    assertFalse(newScore.scores.map(_._2).collect.forall(_ == 0.0))
  }

  @Test
  def testScore(): Unit = sparkTest("testScore") {
    val (coordinate, model) = generateFixedEffectCoordinateAndModel(
      featureShardId, NumTrainingSamples, Dimensionality)

    val score = coordinate.score(model)

    assertEquals(score.scores.count, NumTrainingSamples)
    assertTrue(score.scores.map(_._2).collect.forall(_ == 0.0))
  }
}

object FixedEffectCoordinateTest {
  val Seed = 7
  val NumTrainingSamples = 1000
  val Dimensionality = 10
  val NumPartitions = 4
}
