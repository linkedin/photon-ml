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

import com.linkedin.photon.ml.optimization.{GLMOptimizationConfiguration, OptimizerType, RegularizationType}

/**
 * Some simple tests for GLMOptimizationConfiguration
 */
class GLMOptimizationConfigurationTest {

  import GLMOptimizationConfiguration.{SPLITTER => S}

  @DataProvider
  def invalidStringConfigs(): Array[Array[Any]] = {
    Array(
      Array(s"10${S}1e-2${S}1.0$S-0.2${S}TRON${S}L2"),
      Array(s"10$S${S}1e-2${S}1.0$S-0.2${S}TRON${S}L2"),
      Array(s"10${S}1e-2${S}1.0${S}1.2${S}TRON${S}L2"),
      Array(s"10${S}1e-2${S}1.0${S}0.2${S}f0O${S}L2"),
      Array(s"10${S}1e-2${S}1.0${S}0.2${S}TRON${S}bAR"),
      Array(s"10${S}1e-2${S}1.0${S}0.2${S}TRON"),
      Array(s"10${S}1e-2${S}0.2${S}TRON${S}L2")
    )
  }

  @Test(dataProvider = "invalidStringConfigs",
    expectedExceptions = Array(classOf[NoSuchElementException], classOf[IllegalArgumentException]))
  def testParseAndBuild(configStr: String): Unit = {
    println(GLMOptimizationConfiguration.parseAndBuildFromString(configStr))
  }

  @DataProvider
  def validStringConfigs(): Array[Array[Any]] = {
    Array(
      Array(s"10${S}1e-2${S}1.0${S}0.3${S}TRON${S}L2"),
      // With space before/after the splitters
      Array(s" 10${S}1e-2 $S 1.0 ${S}0.3$S TRON$S L2 ")
    )
  }

  @Test(dataProvider = "validStringConfigs")
  def testParseAndBuildWithValidString(configStr: String): Unit = {
    val config = GLMOptimizationConfiguration.parseAndBuildFromString(configStr)
    val optimizerConfig = config.optimizerConfig
    val regularizationContext = config.regularizationContext

    assertEquals(optimizerConfig.maximumIterations, 10)
    assertEquals(optimizerConfig.tolerance, 1e-2)
    assertEquals(optimizerConfig.optimizerType, OptimizerType.TRON)
    assertEquals(regularizationContext.regularizationType, RegularizationType.L2)
    assertEquals(config.regularizationWeight, 1.0)
    assertEquals(config.downSamplingRate, 0.3)
  }
}
