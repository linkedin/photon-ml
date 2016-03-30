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