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
package com.linkedin.photon.ml.data.avro

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.util.DateRange

/**
 * Unit tests for the [[NameAndTermFeatureBagsDriver]].
 */
class NameAndTermFeatureBagsDriverTest {

  /**
   * Test that a [[ParamMap]] with only required parameters set can be valid input.
   */
  @Test
  def testMinValidParamMap(): Unit = {

    val featureBagKey = "someKey"

    val mockPath = mock(classOf[Path])

    val validParamMap = ParamMap
      .empty
      .put(NameAndTermFeatureBagsDriver.inputDataDirectories, Set[Path](mockPath))
      .put(NameAndTermFeatureBagsDriver.rootOutputDirectory, mockPath)
      .put(NameAndTermFeatureBagsDriver.featureBagsKeys, Set[String](featureBagKey))

    NameAndTermFeatureBagsDriver.validateParams(validParamMap)
  }

  /**
   * Test that a [[ParamMap]] with all parameters set can be valid input.
   */
  @Test
  def testMaxValidParamMap(): Unit = {

    val featureBagKey = "someKey"
    val overrideOutputDirectory = true
    val applicationName = "someName"

    val mockPath = mock(classOf[Path])
    val mockDateRange = mock(classOf[DateRange])

    val validParamMap = ParamMap
      .empty
      .put(NameAndTermFeatureBagsDriver.inputDataDirectories, Set[Path](mockPath))
      .put(NameAndTermFeatureBagsDriver.inputDataDateRange, mockDateRange)
      .put(NameAndTermFeatureBagsDriver.rootOutputDirectory, mockPath)
      .put(NameAndTermFeatureBagsDriver.overrideOutputDirectory, overrideOutputDirectory)
      .put(NameAndTermFeatureBagsDriver.featureBagsKeys, Set[String](featureBagKey))
      .put(NameAndTermFeatureBagsDriver.applicationName, applicationName)

    NameAndTermFeatureBagsDriver.validateParams(validParamMap)
  }

  @DataProvider
  def invalidParamMaps(): Array[Array[Any]] = {

    val featureBagKey = "someKey"

    val mockPath = mock(classOf[Path])

    val validParamMap = ParamMap
      .empty
      .put(NameAndTermFeatureBagsDriver.inputDataDirectories, Set[Path](mockPath))
      .put(NameAndTermFeatureBagsDriver.rootOutputDirectory, mockPath)
      .put(NameAndTermFeatureBagsDriver.featureBagsKeys, Set[String](featureBagKey))

    NameAndTermFeatureBagsDriver.validateParams(validParamMap)

    Array(
      // No input data directories
      Array(validParamMap.copy.remove(NameAndTermFeatureBagsDriver.inputDataDirectories)),
      // No root output directory
      Array(validParamMap.copy.remove(NameAndTermFeatureBagsDriver.rootOutputDirectory)),
      // No feature bags keys
      Array(validParamMap.copy.remove(NameAndTermFeatureBagsDriver.featureBagsKeys)))
  }

  /**
   * Test that invalid parameters will be correctly rejected.
   *
   * @param params A [[ParamMap]] with one or more flaws
   */
  @Test(dataProvider = "invalidParamMaps", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testValidateParams(params: ParamMap): Unit = NameAndTermFeatureBagsDriver.validateParams(params)

  /**
   * Test that default values are set for all parameters that require them.
   */
  @Test
  def testDefaultParams(): Unit = {

    NameAndTermFeatureBagsDriver.clear()

    NameAndTermFeatureBagsDriver.getOrDefault(NameAndTermFeatureBagsDriver.overrideOutputDirectory)
    NameAndTermFeatureBagsDriver.getOrDefault(NameAndTermFeatureBagsDriver.applicationName)
  }

  /**
   * Test that set parameters can be cleared correctly.
   */
  @Test(dependsOnMethods = Array("testDefaultParams"))
  def testClear(): Unit = {

    val mockPath = mock(classOf[Path])

    NameAndTermFeatureBagsDriver.set(NameAndTermFeatureBagsDriver.rootOutputDirectory, mockPath)

    assertEquals(NameAndTermFeatureBagsDriver.get(NameAndTermFeatureBagsDriver.rootOutputDirectory), Some(mockPath))

    NameAndTermFeatureBagsDriver.clear()

    assertEquals(NameAndTermFeatureBagsDriver.get(NameAndTermFeatureBagsDriver.rootOutputDirectory), None)
  }
}
