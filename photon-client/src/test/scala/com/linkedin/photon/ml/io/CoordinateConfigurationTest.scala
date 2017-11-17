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
package com.linkedin.photon.ml.io

import org.mockito.Mockito._
import org.testng.Assert.{assertEquals, assertTrue}
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.{FixedEffectDataConfiguration, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.optimization.{L2RegularizationContext, NoRegularizationContext, OptimizerConfig, RegularizationContext}

/**
 * Unit tests for the derived classes of [[CoordinateConfiguration]].
 */
class CoordinateConfigurationTest {

  /**
   * Test that the [[FixedEffectCoordinateConfiguration]] can correctly expand its
   * [[FixedEffectOptimizationConfiguration]] for each regularization weight.
   */
  @Test
  def testExpandFixedEffect(): Unit = {

    val mockDataConfig = mock(classOf[FixedEffectDataConfiguration])
    val mockOptimizerConfig = mock(classOf[OptimizerConfig])
    val mockRegularizationContext = mock(classOf[RegularizationContext])

    val listRegWeights = Seq(9D, 8D, 7D, 6D, 5D, 4D, 3D, 2D, 1D)

    val baseOptConfig = FixedEffectOptimizationConfiguration(mockOptimizerConfig, mockRegularizationContext)
    val config = FixedEffectCoordinateConfiguration(mockDataConfig, baseOptConfig, listRegWeights.toSet)
    val configsList = config.expandOptimizationConfigurations

    assertEquals(configsList.size, listRegWeights.size)
    configsList
      .map(_.asInstanceOf[FixedEffectOptimizationConfiguration].regularizationWeight)
      .zip(listRegWeights)
      .foreach { case (configWeight, origWeight) =>
        assertEquals(configWeight, origWeight)
      }
  }

  /**
   * Test that the apply helper for [[FixedEffectCoordinateConfiguration]] correctly generates configurations without
   * regularization.
   */
  @Test
  def testConstructorFixedEffect(): Unit = {

    val mockDataConfig = mock(classOf[FixedEffectDataConfiguration])
    val mockOptimizerConfig = mock(classOf[OptimizerConfig])

    val regWeights = Seq(3D, 2D, 1D)

    val optConfig1 = FixedEffectOptimizationConfiguration(mockOptimizerConfig, NoRegularizationContext)
    val optConfig2 = FixedEffectOptimizationConfiguration(mockOptimizerConfig, L2RegularizationContext)

    val config1 = FixedEffectCoordinateConfiguration(mockDataConfig, optConfig1, regWeights.toSet)
    val config2 = FixedEffectCoordinateConfiguration(mockDataConfig, optConfig2, regWeights.toSet)

    assertTrue(config1.expandOptimizationConfigurations.head.equals(optConfig1))
    config2
      .expandOptimizationConfigurations
      .map(_.asInstanceOf[FixedEffectOptimizationConfiguration].regularizationWeight)
      .zip(regWeights)
      .foreach { case (configWeight, origWeight) =>
        assertEquals(configWeight, origWeight)
      }
  }

  /**
   * Test that the [[RandomEffectCoordinateConfiguration]] can correctly expand its
   * [[RandomEffectOptimizationConfiguration]] for each regularization weight.
   */
  @Test
  def testExpandRandomEffect(): Unit = {

    val mockDataConfig = mock(classOf[RandomEffectDataConfiguration])
    val mockOptimizerConfig = mock(classOf[OptimizerConfig])
    val mockRegularizationContext = mock(classOf[RegularizationContext])

    val listRegWeights = Seq(9D, 8D, 7D, 6D, 5D, 4D, 3D, 2D, 1D)

    val baseOptConfig = RandomEffectOptimizationConfiguration(mockOptimizerConfig, mockRegularizationContext)
    val config = RandomEffectCoordinateConfiguration(mockDataConfig, baseOptConfig, listRegWeights.toSet)
    val configsList = config.expandOptimizationConfigurations

    assertEquals(configsList.size, listRegWeights.size)
    configsList
      .map(_.asInstanceOf[RandomEffectOptimizationConfiguration].regularizationWeight)
      .zip(listRegWeights)
      .foreach { case (configWeight, origWeight) =>
        assertEquals(configWeight, origWeight)
      }
  }

  /**
   * Test that the apply helper for [[RandomEffectCoordinateConfiguration]] correctly generates configurations without
   * regularization.
   */
  @Test
  def testConstructorRandomEffect(): Unit = {

    val mockDataConfig = mock(classOf[RandomEffectDataConfiguration])
    val mockOptimizerConfig = mock(classOf[OptimizerConfig])

    val regWeights = Seq(3D, 2D, 1D)

    val optConfig1 = RandomEffectOptimizationConfiguration(mockOptimizerConfig, NoRegularizationContext)
    val optConfig2 = RandomEffectOptimizationConfiguration(mockOptimizerConfig, L2RegularizationContext)

    val config1 = RandomEffectCoordinateConfiguration(mockDataConfig, optConfig1, regWeights.toSet)
    val config2 = RandomEffectCoordinateConfiguration(mockDataConfig, optConfig2, regWeights.toSet)

    assertTrue(config1.expandOptimizationConfigurations.head.equals(optConfig1))
    config2
      .expandOptimizationConfigurations
      .map(_.asInstanceOf[RandomEffectOptimizationConfiguration].regularizationWeight)
      .zip(regWeights)
      .foreach { case (configWeight, origWeight) =>
        assertEquals(configWeight, origWeight)
      }
  }
}
