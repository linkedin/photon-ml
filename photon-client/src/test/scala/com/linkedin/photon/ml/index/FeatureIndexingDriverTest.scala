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
package com.linkedin.photon.ml.index

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.util.DateRange

/**
 * Unit tests for [[FeatureIndexingDriver]].
 */
class FeatureIndexingDriverTest {

  /**
   * Test that a [[ParamMap]] with only required parameters set can be valid input.
   */
  @Test
  def testMinValidParamMap(): Unit = {

    val featureShardId = "id"
    val numPartitions = 123

    val mockPath = mock(classOf[Path])
    val mockFeatureShardConfig = mock(classOf[FeatureShardConfiguration])

    doReturn(false).when(mockFeatureShardConfig).hasIntercept

    val validParamMap = ParamMap
      .empty
      .put(FeatureIndexingDriver.inputDataDirectories, Set[Path](mockPath))
      .put(FeatureIndexingDriver.rootOutputDirectory, mockPath)
      .put(FeatureIndexingDriver.numPartitions, numPartitions)
      .put(FeatureIndexingDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))

    FeatureIndexingDriver.validateParams(validParamMap)
  }

  /**
   * Test that a [[ParamMap]] with all parameters set can be valid input.
   */
  @Test
  def testMaxValidParamMap(): Unit = {

    val featureShardId = "id"
    val numPartitions = 123
    val overrideOutputDir = true
    val applicationName = "someName"

    val mockPath = mock(classOf[Path])
    val mockDateRange = mock(classOf[DateRange])
    val mockFeatureShardConfig = mock(classOf[FeatureShardConfiguration])

    doReturn(false).when(mockFeatureShardConfig).hasIntercept

    val validParamMap = ParamMap
      .empty
      .put(FeatureIndexingDriver.inputDataDirectories, Set[Path](mockPath))
      .put(FeatureIndexingDriver.inputDataDateRange, mockDateRange)
      .put(FeatureIndexingDriver.minInputPartitions, numPartitions)
      .put(FeatureIndexingDriver.rootOutputDirectory, mockPath)
      .put(FeatureIndexingDriver.overrideOutputDirectory, overrideOutputDir)
      .put(FeatureIndexingDriver.numPartitions, numPartitions)
      .put(FeatureIndexingDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))
      .put(FeatureIndexingDriver.applicationName, applicationName)

    FeatureIndexingDriver.validateParams(validParamMap)
  }

  @DataProvider
  def invalidParamMaps(): Array[Array[Any]] = {

  val featureShardId = "id"
  val numPartitions = 123

  val mockPath = mock(classOf[Path])
  val mockFeatureShardConfig = mock(classOf[FeatureShardConfiguration])

  doReturn(false).when(mockFeatureShardConfig).hasIntercept

  val validParamMap = ParamMap
    .empty
    .put(FeatureIndexingDriver.inputDataDirectories, Set[Path](mockPath))
    .put(FeatureIndexingDriver.rootOutputDirectory, mockPath)
    .put(FeatureIndexingDriver.numPartitions, numPartitions)
    .put(FeatureIndexingDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))

    Array(
      // No input data directories
      Array(validParamMap.copy.remove(FeatureIndexingDriver.inputDataDirectories)),
      // No root output directory
      Array(validParamMap.copy.remove(FeatureIndexingDriver.rootOutputDirectory)),
      // No num partitions
      Array(validParamMap.copy.remove(FeatureIndexingDriver.numPartitions)),
      // No feature shard configurations
      Array(validParamMap.copy.remove(FeatureIndexingDriver.featureShardConfigurations)))
  }

  /**
   * Test that invalid parameters will be correctly rejected.
   *
   * @param params A [[ParamMap]] with one or more flaws
   */
  @Test(dataProvider = "invalidParamMaps", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testValidateParams(params: ParamMap): Unit = FeatureIndexingDriver.validateParams(params)

  /**
   * Test that default values are set for all parameters that require them.
   */
  @Test
  def testDefaultParams(): Unit = {

    FeatureIndexingDriver.clear()

    FeatureIndexingDriver.getOrDefault(FeatureIndexingDriver.minInputPartitions)
    FeatureIndexingDriver.getOrDefault(FeatureIndexingDriver.overrideOutputDirectory)
    FeatureIndexingDriver.getOrDefault(FeatureIndexingDriver.applicationName)
  }

  /**
   * Test that set parameters can be cleared correctly.
   */
  @Test
  def testClear(): Unit = {

    val mockPath = mock(classOf[Path])

    FeatureIndexingDriver.set(FeatureIndexingDriver.rootOutputDirectory, mockPath)

    assertEquals(FeatureIndexingDriver.get(FeatureIndexingDriver.rootOutputDirectory), Some(mockPath))

    FeatureIndexingDriver.clear()

    assertEquals(FeatureIndexingDriver.get(FeatureIndexingDriver.rootOutputDirectory), None)
  }
}
