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
package com.linkedin.photon.ml.cli.game

import scala.util.{Failure, Success, Try}

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.mockito.Mockito._
import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.DataValidationType
import com.linkedin.photon.ml.data.InputColumnsNames
import com.linkedin.photon.ml.evaluation.EvaluatorType
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.util.{DateRange, PhotonLogger}

/**
 * Unit tests for [[GameDriver]].
 */
class GameDriverTest {

  import GameDriverTest._

  /**
   * Test that required parameters must have an explicitly set value.
   */
  @Test
  def testGetRequiredParam(): Unit = {

    val mockPath = mock(classOf[Path])

    MockGameDriver.set(MockGameDriver.rootOutputDirectory, mockPath)
    MockGameDriver.mockSetDefault(MockGameDriver.inputDataDirectories, Set(mockPath))

    assertEquals(MockGameDriver.getDefault(MockGameDriver.rootOutputDirectory), None)
    assertEquals(MockGameDriver.get(MockGameDriver.rootOutputDirectory), Some(mockPath))
    assertEquals(MockGameDriver.getRequiredParam(MockGameDriver.rootOutputDirectory), mockPath)

    assertEquals(MockGameDriver.getDefault(MockGameDriver.featureShardConfigurations), None)
    assertEquals(MockGameDriver.get(MockGameDriver.featureShardConfigurations), None)
    Try(MockGameDriver.getRequiredParam(MockGameDriver.featureShardConfigurations)) match {
      case Success(_) => assert(false)
      case Failure(_) =>
    }

    assertEquals(MockGameDriver.getDefault(MockGameDriver.inputDataDirectories), Some(Set(mockPath)))
    assertEquals(MockGameDriver.get(MockGameDriver.inputDataDirectories), None)
    Try(MockGameDriver.getRequiredParam(MockGameDriver.inputDataDirectories)) match {
      case Success(_) => assert(false)
      case Failure(_) =>
    }
  }

  /**
   * Test that a [[ParamMap]] with only required parameters set can be valid input.
   */
  @Test
  def testMinValidParamMap(): Unit = {

    val featureShardId = "id"

    val mockPath = mock(classOf[Path])
    val mockFeatureShardConfig = mock(classOf[FeatureShardConfiguration])

    doReturn(false).when(mockFeatureShardConfig).hasIntercept

    val validParamMap = ParamMap
      .empty
      .put(MockGameDriver.inputDataDirectories, Set[Path](mockPath))
      .put(MockGameDriver.rootOutputDirectory, mockPath)
      .put(MockGameDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))

    MockGameDriver.validateParams(validParamMap)
  }

  /**
   * Test that a [[ParamMap]] with all parameters set can be valid input.
   */
  @Test
  def testMaxValidParamMap(): Unit = {

    val featureShardId = "id"

    val mockBoolean = true
    val mockInt = 10
    val mockString = "text"

    val mockPath = mock(classOf[Path])
    val mockDateRange = mock(classOf[DateRange])
    val mockInputColumnNames = mock(classOf[InputColumnsNames])
    val mockEvaluatorType = mock(classOf[EvaluatorType])
    val mockFeatureShardConfig = mock(classOf[FeatureShardConfiguration])

    doReturn(true).when(mockFeatureShardConfig).hasIntercept

    val validParamMap = ParamMap
      .empty
      .put(MockGameDriver.inputDataDirectories, Set[Path](mockPath))
      .put(MockGameDriver.inputDataDateRange, mockDateRange)
      .put(MockGameDriver.offHeapIndexMapDirectory, mockPath)
      .put(MockGameDriver.offHeapIndexMapPartitions, mockInt)
      .put(MockGameDriver.inputColumnNames, mockInputColumnNames)
      .put(MockGameDriver.evaluators, Seq[EvaluatorType](mockEvaluatorType))
      .put(MockGameDriver.rootOutputDirectory, mockPath)
      .put(MockGameDriver.overrideOutputDirectory, mockBoolean)
      .put(MockGameDriver.outputFilesLimit, mockInt)
      .put(MockGameDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))
      .put(MockGameDriver.dataValidation, DataValidationType.VALIDATE_FULL)
      .put(MockGameDriver.logLevel, mockInt)
      .put(MockGameDriver.applicationName, mockString)

    MockGameDriver.validateParams(validParamMap)
  }

  @DataProvider
  def invalidParamMaps(): Array[Array[Any]] = {

    val featureShardId = "id"

    val mockPath = mock(classOf[Path])
    val mockFeatureShardConfig = mock(classOf[FeatureShardConfiguration])

    doReturn(false).when(mockFeatureShardConfig).hasIntercept

    val validParamMap = ParamMap
      .empty
      .put(MockGameDriver.inputDataDirectories, Set[Path](mockPath))
      .put(MockGameDriver.rootOutputDirectory, mockPath)
      .put(MockGameDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))

    Array(
      // No input data directories
      Array(validParamMap.copy.remove(MockGameDriver.inputDataDirectories)),
      // No root output directory
      Array(validParamMap.copy.remove(MockGameDriver.rootOutputDirectory)),
      // No feature shard configurations
      Array(validParamMap.copy.remove(MockGameDriver.featureShardConfigurations)),
      // Off-heap map dir without partitions
      Array(validParamMap.copy.put(MockGameDriver.offHeapIndexMapDirectory, mockPath)),
      // Off-heap map partitions without dir
      Array(validParamMap.copy.put(MockGameDriver.offHeapIndexMapPartitions, 1)),
      // Both off-heap map and features directory
      Array(
        validParamMap
          .copy
          .put(MockGameDriver.offHeapIndexMapDirectory, mockPath)
          .put(MockGameDriver.featureBagsDirectory, mockPath)))
  }

  /**
   * Test that invalid parameters will be correctly rejected.
   *
   * @param params A [[ParamMap]] with one or more flaws
   */
  @Test(dataProvider = "invalidParamMaps", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testValidateParams(params: ParamMap): Unit = {

    MockGameDriver.clear()

    params.toSeq.foreach(pair => MockGameDriver.set(pair.param.asInstanceOf[Param[Any]], pair.value))
    MockGameDriver.validateParams()
  }
}

object GameDriverTest {

  /**
   * Mock [[GameDriver]] for testing trait functions (including protected functions).
   */
  object MockGameDriver extends GameDriver {

    var sc: SparkContext = _
    implicit var logger: PhotonLogger = _

    def clear(): Unit = params.foreach(MockGameDriver.clear)
    def copy(extra: ParamMap): Params = this

    override def getRequiredParam[T](param: Param[T]): T = super.getRequiredParam(param)
    def mockSetDefault[T](param: Param[T], value: T): Unit = {
      setDefault(param, value)
    }
  }
}
