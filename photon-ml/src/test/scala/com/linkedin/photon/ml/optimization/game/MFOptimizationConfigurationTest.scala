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
package com.linkedin.photon.ml.optimization.game

import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.optimization.GLMOptimizationConfiguration

/**
 * Some simple tests for MFOptimizationConfiguration
 */
class MFOptimizationConfigurationTest {

  import MFOptimizationConfiguration.{SPLITTER => S}

  @DataProvider
  def invalidStringConfigs(): Array[Array[Any]] = {
    Array(
      Array(s"NotANumber${S}10"),
      Array(s"5d${S}10"),
      Array(s"5${S}NotANumber"),
      Array(s"5")
    )
  }

  @Test(dataProvider = "invalidStringConfigs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testParseAndBuild(configStr: String): Unit = {
    println(GLMOptimizationConfiguration.parseAndBuildFromString(configStr))
  }

  @DataProvider
  def validStringConfigs(): Array[Array[Any]] = {
    Array(
      Array(s"10${S}20"),
      // With space before/after the splitters
      Array(s" 10   $S  20  ")
    )
  }

  @Test(dataProvider = "validStringConfigs")
  def testParseAndBuildWithValidString(configStr: String): Unit = {
    val config = MFOptimizationConfiguration.parseAndBuildFromString(configStr)
    assertEquals(config.maxNumberIterations, 10)
    assertEquals(config.numFactors, 20)
  }
}
