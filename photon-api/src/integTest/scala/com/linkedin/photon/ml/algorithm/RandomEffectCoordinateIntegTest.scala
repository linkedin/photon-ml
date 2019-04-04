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

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.model.RandomEffectModel
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.{GameTestUtils, MathUtils}

/**
 * Integration tests for [[RandomEffectCoordinate]].
 */
class RandomEffectCoordinateIntegTest extends SparkTestUtils with GameTestUtils {

  import RandomEffectCoordinateIntegTest._

  @DataProvider
  def numEntitiesDataProvider(): Array[Array[Any]] = Array(Array(1), Array(2), Array(10))

  /**
   * Test that a [[RandomEffectCoordinate]] can train a new model.
   *
   * @param numEntities The number of unique per-entity models to train
   */
  @Test(dataProvider = "numEntitiesDataProvider", dependsOnMethods = Array("testScore"))
  def testTrainModel(numEntities: Int): Unit = sparkTest("testUpdateModel") {

    val (coordinate, model) = generateRandomEffectCoordinateAndModel(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_ID,
      numEntities,
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
   * Test that a [[RandomEffectCoordinate]] can train a new model, using an existing [[RandomEffectModel]] as a starting
   * point for optimization, and retain existing models for which new data does not exist.
   */
  @Test
  def testTrainWithExtraModel(): Unit = sparkTest("testUpdateInitialModel") {

    val extraREID = "reExtra"
    val (coordinate, baseModel) = generateRandomEffectCoordinateAndModel(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_ID,
      numEntities = 1,
      NUM_TRAINING_SAMPLES,
      DIMENSIONALITY)

    // Add in an item that exists in the prior model, but not the data
    val randomEffectIds = baseModel.modelsRDD.keys.collect() :+ extraREID
    val extraModel =
      baseModel.update(sc.parallelize(generateLinearModelsForRandomEffects(randomEffectIds, DIMENSIONALITY)))
    val (newModel, _) = coordinate.trainModel(extraModel)

    newModel match {
      case randomEffectModel: RandomEffectModel =>
        // Make sure that the prior model items are still there
        assertEquals(randomEffectModel.modelsRDD.map(_._1).collect.toSet, randomEffectIds.toSet)

      case other =>
        fail(s"Unexpected model type: ${other.getClass.getName}")
    }
  }

  /**
   * Test that a [[RandomEffectCoordinate]] can score data using a [[RandomEffectModel]].
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

    assertEquals(score.scores.count, numEntities * NUM_TRAINING_SAMPLES)
    assertTrue(score.scores.map(_._2).collect.forall(MathUtils.isAlmostZero))
  }
}

object RandomEffectCoordinateIntegTest {

  private val RANDOM_EFFECT_TYPE = "random-effect-1"
  private val FEATURE_SHARD_ID = "shard1"
  private val NUM_TRAINING_SAMPLES = 1000
  private val DIMENSIONALITY = 10
}
