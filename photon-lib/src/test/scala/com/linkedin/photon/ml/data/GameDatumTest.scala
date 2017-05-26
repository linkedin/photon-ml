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

import java.util.NoSuchElementException

import breeze.linalg.Vector
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

/**
 * Unit test cases for the [[GameDatum]]
 */
class GameDatumTest {

  import GameDatumTest._

  private val gameDatumWithEverything = new GameDatum(
    response = 1.0,
    offsetOpt = Some(-10.0),
    weightOpt = Some(5.0),
    featureShardContainer = Map(DEFAULT_SHARD_ID -> Vector.zeros[Double](1)),
    idTagToValueMap = Map("foo" -> "bar"))

  private val gameDatumWithoutOffset = new GameDatum(
    response = 1.0,
    offsetOpt = None,
    weightOpt = Some(5.0),
    featureShardContainer = Map(DEFAULT_SHARD_ID -> Vector.zeros[Double](1)),
    idTagToValueMap = Map("foo" -> "bar"))

  private val gameDatumWithoutWeight = new GameDatum(
    response = 1.0,
    offsetOpt = Some(-10.0),
    weightOpt = None,
    featureShardContainer = Map(DEFAULT_SHARD_ID -> Vector.zeros[Double](1)),
    idTagToValueMap = Map("uid" -> "uid"))

  private val gameDatumWithoutIdTypeToValueMap = new GameDatum(
    response = 1.0,
    offsetOpt = Some(-10.0),
    weightOpt = Some(5.0),
    featureShardContainer = Map(DEFAULT_SHARD_ID -> Vector.zeros[Double](1)),
    idTagToValueMap = Map())

  private val gameDatumWithResponseAndFeatures = new GameDatum(
    response = 1.0,
    offsetOpt = None,
    weightOpt = None,
    featureShardContainer = Map(DEFAULT_SHARD_ID -> Vector.zeros[Double](1)),
    idTagToValueMap = Map())

  /**
   * Provide constructed [[GameDatum]] as testing input.
   */
  @DataProvider
  def gameDatumDataProvider(): Array[Array[Any]] = {
    Array(
      Array(gameDatumWithEverything),
      Array(gameDatumWithoutOffset),
      Array(gameDatumWithoutWeight),
      Array(gameDatumWithoutIdTypeToValueMap),
      Array(gameDatumWithResponseAndFeatures))
  }

  /**
   * Test that [[GameDatum]] can correctly construct a [[LabeledPoint]] for a known feature shard.
   */
  @Test(dataProvider = "gameDatumDataProvider")
  def generateLabeledPointWithFeatureShardIdTest(gameDatum: GameDatum): Unit = {

    val expectedLabeledPoint = new LabeledPoint(
      label = gameDatum.response,
      features = gameDatum.featureShardContainer(DEFAULT_SHARD_ID),
      offset = gameDatum.offset,
      weight = gameDatum.weight)
    val actualLabeledPoint = gameDatum.generateLabeledPointWithFeatureShardId(DEFAULT_SHARD_ID)

    assertEquals(actualLabeledPoint.label, expectedLabeledPoint.label)
    assertEquals(actualLabeledPoint.features, expectedLabeledPoint.features)
    assertEquals(actualLabeledPoint.offset, expectedLabeledPoint.offset)
    assertEquals(actualLabeledPoint.weight, expectedLabeledPoint.weight)
  }

  /**
   * Test that [[GameDatum]] throws exception when attempting to construct a [[LabeledPoint]] for an unknown feature
   * shard.
   */
  @Test(expectedExceptions = Array(classOf[NoSuchElementException]))
  def generateLabeledPointWithNonExistentFeatureShardIdTest(): Unit = {
    gameDatumWithResponseAndFeatures.generateLabeledPointWithFeatureShardId("unknownShardId")
  }
}

object GameDatumTest {
  private val DEFAULT_SHARD_ID = "shardId"
}
