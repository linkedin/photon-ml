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

import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.projector.{IdentityProjection, IndexMapProjection, RandomProjection}

/**
 *
 */
class RandomEffectDataConfigurationTest {

  import RandomEffectDataConfiguration.{FIRST_LEVEL_SPLITTER => F, SECOND_LEVEL_SPLITTER => S}

  @DataProvider
  def invalidStringConfigs(): Array[Array[Any]] = {
    Array(
      Array(s"randomEffectType${F}featureShardId${F}1${F}10${F}5${F}20${F}randomNoLatentFactor"),
      Array(s"randomEffectType${F}featureShardId${F}1${F}10${F}5${F}20${F}randomMissingSplitter5"),
      Array(s"missOneConfig${F}1${F}10${F}5${F}20${F}random=5"),
      Array(s"randomEffectType${F}featureShardId${F}notANumber${F}10${F}5${F}20${F}random${S}5"),
      Array(s"randomEffectType${F}featureShardId${F}1${F}10${F}5${F}notANumber${F}index_map"),
      Array(s"randomEffectType${F}featureShardId${F}1${F}10${F}notANumber${F}20${F}identity"),
      Array(s"randomEffectType${F}featureShardId${F}1${F}notANumber${F}5${F}20${F}identity"),
      Array(s"randomEffectType${F}featureShardId${F}1${F}10${F}5${F}20${F}unknownProjector")
    )
  }

  @Test(dataProvider = "invalidStringConfigs",
    expectedExceptions = Array(classOf[IllegalArgumentException], classOf[NoSuchElementException]))
  def testParseAndBuild(configStr: String): Unit = {
    println(RandomEffectDataConfiguration.parseAndBuildFromString(configStr))
  }

  @DataProvider
  def validStringConfigs(): Array[Array[Any]] = {
    Array(
      Array(s"randomEffectType${F}featureShardId${F}1${F}10${F}5${F}20d${F}random${S}5"),
      Array(s"randomEffectType${F}featureShardId${F}1${F}10${F}5${F}20d${F}index_map"),
      Array(s"randomEffectType${F}featureShardId${F}1${F}10${F}5${F}20d${F}identity"),
      // With space before/after the splitters
      Array(s"randomEffectType   $F    featureShardId  $F   1  $F  10 $F 5 $F 20d$F random   $S  5")
    )
  }

  @Test(dataProvider = "validStringConfigs")
  def testParseAndBuildWithValidString(configStr: String): Unit = {
    val config = RandomEffectDataConfiguration.parseAndBuildFromString(configStr)
    assertEquals(config.randomEffectType, "randomEffectType")
    assertEquals(config.featureShardId, "featureShardId")
    assertEquals(config.numPartitions, 1)
    assertEquals(config.numActiveDataPointsToKeepUpperBound, 10)
    assertEquals(config.numPassiveDataPointsToKeepLowerBound, 5)
    assertEquals(config.numFeaturesToSamplesRatioUpperBound, 20d)

    if (configStr.endsWith("5")) {
      assertEquals(config.projectorType, RandomProjection(5))
    } else if (configStr.endsWith("index_map")) {
      assertEquals(config.projectorType, IndexMapProjection)
    } else {
      assertEquals(config.projectorType, IdentityProjection)
    }
  }
}
