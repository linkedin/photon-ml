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
import com.linkedin.photon.ml.optimization.game.{FactoredRandomEffectOptimizationConfiguration, FixedEffectOptimizationConfiguration, MFOptimizationConfiguration, RandomEffectOptimizationConfiguration}
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

    val baseRegWeight = 1D
    val listRegWeights = Seq(9D, 8D, 7D, 6D, 5D, 4D, 3D, 2D, 1D)

    val baseOptConfig = FixedEffectOptimizationConfiguration(
      mockOptimizerConfig,
      mockRegularizationContext,
      baseRegWeight)

    val config1 = FixedEffectCoordinateConfiguration(mockDataConfig, baseOptConfig)
    val config2 = FixedEffectCoordinateConfiguration(mockDataConfig, baseOptConfig, listRegWeights.toSet)
    val configsList1 = config1.expandOptimizationConfigurations
    val configsList2 = config2.expandOptimizationConfigurations

    assertEquals(configsList1.size, 1)
    assertTrue(configsList1.head.eq(baseOptConfig))

    assertEquals(configsList2.size, listRegWeights.size)
    configsList2
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

    assertTrue(config1.expandOptimizationConfigurations.head.eq(optConfig1))
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

    val baseRegWeight = 1D
    val listRegWeights = Seq(9D, 8D, 7D, 6D, 5D, 4D, 3D, 2D, 1D)

    val baseOptConfig = RandomEffectOptimizationConfiguration(
      mockOptimizerConfig,
      mockRegularizationContext,
      baseRegWeight)

    val config1 = RandomEffectCoordinateConfiguration(mockDataConfig, baseOptConfig)
    val config2 = RandomEffectCoordinateConfiguration(mockDataConfig, baseOptConfig, listRegWeights.toSet)
    val configsList1 = config1.expandOptimizationConfigurations
    val configsList2 = config2.expandOptimizationConfigurations

    assertEquals(configsList1.size, 1)
    assertTrue(configsList1.head.eq(baseOptConfig))

    assertEquals(configsList2.size, listRegWeights.size)
    configsList2
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

    assertTrue(config1.expandOptimizationConfigurations.head.eq(optConfig1))
    config2
      .expandOptimizationConfigurations
      .map(_.asInstanceOf[RandomEffectOptimizationConfiguration].regularizationWeight)
      .zip(regWeights)
      .foreach { case (configWeight, origWeight) =>
        assertEquals(configWeight, origWeight)
      }
  }

  /**
   * Test that the [[FactoredRandomEffectCoordinateConfiguration]] can correctly expand its
   * [[FactoredRandomEffectOptimizationConfiguration]] for each combination of regularization weights.
   */
  @Test
  def testExpandFactoredRandomEffect(): Unit = {

    val mockDataConfig = mock(classOf[RandomEffectDataConfiguration])
    val mockMFOptConfig = mock(classOf[MFOptimizationConfiguration])
    val mockOptimizerConfig = mock(classOf[OptimizerConfig])
    val mockRegularizationContext = mock(classOf[RegularizationContext])

    val baseRERegWeight = 1D
    val baseLFRegWeight = 1D
    val listRERegWeights = Seq(2D, 1D)
    val listLFRegWeights = Seq(2D, 1D)
    val listRegWeightPairs = Seq((2D, 2D), (2D, 1D), (1D, 2D), (1D, 1D))

    val baseREOptConfig = RandomEffectOptimizationConfiguration(
      mockOptimizerConfig,
      mockRegularizationContext,
      baseRERegWeight)
    val baseLFOptConfig = RandomEffectOptimizationConfiguration(
      mockOptimizerConfig,
      mockRegularizationContext,
      baseLFRegWeight)
    val baseOptConfig = FactoredRandomEffectOptimizationConfiguration(baseREOptConfig, baseLFOptConfig, mockMFOptConfig)

    val config1 = FactoredRandomEffectCoordinateConfiguration(mockDataConfig, baseOptConfig)
    val config2 = FactoredRandomEffectCoordinateConfiguration(
      mockDataConfig,
      baseOptConfig,
      listRERegWeights.toSet,
      listLFRegWeights.toSet)
    val configsList1 = config1.expandOptimizationConfigurations
    val configsList2 = config2.expandOptimizationConfigurations

    assertEquals(configsList1.size, 1)
    assertTrue(configsList1.head.eq(baseOptConfig))

    assertEquals(configsList2.size, listRegWeightPairs.size)
    configsList2
      .map { optConfig =>
        val fREOptConfig = optConfig.asInstanceOf[FactoredRandomEffectOptimizationConfiguration]

        (fREOptConfig.reOptConfig.regularizationWeight, fREOptConfig.lfOptConfig.regularizationWeight)
      }
      .zip(listRegWeightPairs)
      .foreach { case ((configREWeight, configLFWeight), (origREWeight, origLFWeight)) =>
        assertEquals(configREWeight, origREWeight)
        assertEquals(configLFWeight, origLFWeight)
      }
  }

  /**
   * Test that the apply helper for [[FactoredRandomEffectCoordinateConfiguration]] correctly generates configurations
   * without regularization.
   */
  @Test
  def testConstructorFactoredRandomEffect(): Unit = {

    val mockDataConfig = mock(classOf[RandomEffectDataConfiguration])
    val mockMFOptConfig = mock(classOf[MFOptimizationConfiguration])
    val mockOptimizerConfig = mock(classOf[OptimizerConfig])

    val baseRERegWeight = 1D
    val baseLFRegWeight = 1D
    val listRERegWeights = Set(2D)
    val listLFRegWeights = Set(2D)

    val noRegREOptConfig = RandomEffectOptimizationConfiguration(
      mockOptimizerConfig,
      NoRegularizationContext,
      baseRERegWeight)
    val l2RegREOptConfig = RandomEffectOptimizationConfiguration(
      mockOptimizerConfig,
      L2RegularizationContext,
      baseRERegWeight)
    val noRegLFOptConfig = RandomEffectOptimizationConfiguration(
      mockOptimizerConfig,
      NoRegularizationContext,
      baseLFRegWeight)
    val l2RegLFOptConfig = RandomEffectOptimizationConfiguration(
      mockOptimizerConfig,
      L2RegularizationContext,
      baseLFRegWeight)

    val noRegOptConfig = FactoredRandomEffectOptimizationConfiguration(
      noRegREOptConfig,
      noRegLFOptConfig,
      mockMFOptConfig)
    val rERegOptConfig = FactoredRandomEffectOptimizationConfiguration(
      l2RegREOptConfig,
      noRegLFOptConfig,
      mockMFOptConfig)
    val lFRegOptConfig = FactoredRandomEffectOptimizationConfiguration(
      noRegREOptConfig,
      l2RegLFOptConfig,
      mockMFOptConfig)
    val bothRegOptConfig = FactoredRandomEffectOptimizationConfiguration(
      l2RegREOptConfig,
      l2RegLFOptConfig,
      mockMFOptConfig)

    val noRegWeightPair = (1D, 1D)
    val rERegWeightPair = (2D, 1D)
    val lFRegWeightPair = (1D, 2D)
    val bothRegWeightPair = (2D, 2D)

    val noRegConfigList =
      FactoredRandomEffectCoordinateConfiguration(
        mockDataConfig,
        noRegOptConfig,
        listRERegWeights,
        listLFRegWeights)
      .expandOptimizationConfigurations
    val rERegConfigList =
      FactoredRandomEffectCoordinateConfiguration(
        mockDataConfig,
        rERegOptConfig,
        listRERegWeights,
        listLFRegWeights)
    .expandOptimizationConfigurations
    val lFRegConfigList =
      FactoredRandomEffectCoordinateConfiguration(
        mockDataConfig,
        lFRegOptConfig,
        listRERegWeights,
        listLFRegWeights)
      .expandOptimizationConfigurations
    val bothRegConfigList =
      FactoredRandomEffectCoordinateConfiguration(
        mockDataConfig,
        bothRegOptConfig,
        listRERegWeights,
        listLFRegWeights)
      .expandOptimizationConfigurations

    assertEquals(noRegConfigList.size, 1)
    assertEquals(rERegConfigList.size, 1)
    assertEquals(lFRegConfigList.size, 1)
    assertEquals(bothRegConfigList.size, 1)

    val noRegConfigPair = noRegConfigList
      .map { optConfig =>
        val fREOptConfig = optConfig.asInstanceOf[FactoredRandomEffectOptimizationConfiguration]

        (fREOptConfig.reOptConfig.regularizationWeight, fREOptConfig.lfOptConfig.regularizationWeight)
      }
      .head
    val rERegConfigPair = rERegConfigList
      .map { optConfig =>
        val fREOptConfig = optConfig.asInstanceOf[FactoredRandomEffectOptimizationConfiguration]

        (fREOptConfig.reOptConfig.regularizationWeight, fREOptConfig.lfOptConfig.regularizationWeight)
      }
      .head
    val lFRegConfigPair = lFRegConfigList
      .map { optConfig =>
        val fREOptConfig = optConfig.asInstanceOf[FactoredRandomEffectOptimizationConfiguration]

        (fREOptConfig.reOptConfig.regularizationWeight, fREOptConfig.lfOptConfig.regularizationWeight)
      }
      .head
    val bothRegConfigPair = bothRegConfigList
      .map { optConfig =>
        val fREOptConfig = optConfig.asInstanceOf[FactoredRandomEffectOptimizationConfiguration]

        (fREOptConfig.reOptConfig.regularizationWeight, fREOptConfig.lfOptConfig.regularizationWeight)
      }
      .head

    assertEquals(noRegConfigPair, noRegWeightPair)
    assertEquals(rERegConfigPair, rERegWeightPair)
    assertEquals(lFRegConfigPair, lFRegWeightPair)
    assertEquals(bothRegConfigPair, bothRegWeightPair)
  }
}
