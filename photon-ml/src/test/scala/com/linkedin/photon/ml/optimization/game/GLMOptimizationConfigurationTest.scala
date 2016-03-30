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

import com.linkedin.photon.ml.optimization.{RegularizationType, OptimizerType}
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

/**
 * @author nkatariy
 */
class GLMOptimizationConfigurationTest {
  @DataProvider
  def invalidStringConfigs(): Array[Array[Any]] = {
    Array(
      Array("10,1e-2,1.0,-0.2,TRON,L2"),
      Array("10,1e-2,1.0,1.2,TRON,L2"),
      Array("10,1e-2,1.0,0.2,f0O,L2"),
      Array("10,1e-2,1.0,0.2,TRON,bAR")
      // TODO We get number format exceptions if there are spaces before / after commas. Not adding tests because more robust design is the right solution here
    )
  }

  @Test(dataProvider = "invalidStringConfigs",
    expectedExceptions = Array(classOf[NoSuchElementException], classOf[AssertionError]))
  def testParseAndBuild(configStr: String) = {
    println(GLMOptimizationConfiguration.parseAndBuildFromString(configStr))
  }

  @Test
  def testParseAndBuildWithValidString() = {
    val config = GLMOptimizationConfiguration.parseAndBuildFromString("10,1e-2,1.0,0.3,TRON,L2")
    Assert.assertEquals(config.maxNumberIterations, 10)
    Assert.assertEquals(config.convergenceTolerance, 1e-2)
    Assert.assertEquals(config.regularizationWeight, 1.0)
    Assert.assertEquals(config.downSamplingRate, 0.3)
    Assert.assertEquals(config.optimizerType, OptimizerType.TRON)
    Assert.assertEquals(config.regularizationType, RegularizationType.L2)
  }
}