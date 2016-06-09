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
package com.linkedin.photon.ml.data

import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

class FixedEffectDataConfigurationTest {

  import FixedEffectDataConfiguration.{SPLITTER => S}

  @DataProvider
  def invalidStringConfigs(): Array[Array[Any]] = {
    Array(
      Array(s"shard1${S}5${S}shard2"),
      Array(s"shard1"),
      Array(s"5"),
      Array(s"shard1${S}shard2")
    )
  }

  @Test(dataProvider = "invalidStringConfigs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testParseAndBuild(configStr: String): Unit = {
    println(FixedEffectDataConfiguration.parseAndBuildFromString(configStr))
  }

  @DataProvider
  def validStringConfigs(): Array[Array[Any]] = {
    Array(
      Array(s"shardId${S}1"),
      // With space before/after the splitters
      Array(s" shardId   $S  1    ")
    )
  }

  @Test(dataProvider = "validStringConfigs")
  def testParseAndBuildWithValidString(configStr: String): Unit = {
    val config = FixedEffectDataConfiguration.parseAndBuildFromString(configStr)
    Assert.assertEquals(config.featureShardId, "shardId")
    Assert.assertEquals(config.minNumPartitions, 1)
  }
}
