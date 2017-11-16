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
package com.linkedin.photon.ml.cli.game.scoring

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.{Param, ParamMap}
import org.mockito.Mockito._
import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.DataValidationType
import com.linkedin.photon.ml.data.InputColumnsNames
import com.linkedin.photon.ml.evaluation.EvaluatorType
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.util.DateRange

/**
 * Unit tests for [[GameScoringDriver]].
 */
class GameScoringDriverTest {

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
      .put(GameScoringDriver.inputDataDirectories, Set[Path](mockPath))
      .put(GameScoringDriver.rootOutputDirectory, mockPath)
      .put(GameScoringDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))
      .put(GameScoringDriver.modelInputDirectory, mockPath)

    GameScoringDriver.validateParams(validParamMap)
  }

  /**
   * Test that a [[ParamMap]] with all parameters set can be valid input.
   */
  @Test
  def testMaxValidParamMap(): Unit = {

    val featureShardId = "id"
    val modelId = "someModel"

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
      .put(GameScoringDriver.inputDataDirectories, Set[Path](mockPath))
      .put(GameScoringDriver.inputDataDateRange, mockDateRange)
      .put(GameScoringDriver.offHeapIndexMapDirectory, mockPath)
      .put(GameScoringDriver.offHeapIndexMapPartitions, mockInt)
      .put(GameScoringDriver.inputColumnNames, mockInputColumnNames)
      .put(GameScoringDriver.evaluators, Seq[EvaluatorType](mockEvaluatorType))
      .put(GameScoringDriver.rootOutputDirectory, mockPath)
      .put(GameScoringDriver.overrideOutputDirectory, mockBoolean)
      .put(GameScoringDriver.outputFilesLimit, mockInt)
      .put(GameScoringDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))
      .put(GameScoringDriver.dataValidation, DataValidationType.VALIDATE_FULL)
      .put(GameScoringDriver.logLevel, mockInt)
      .put(GameScoringDriver.applicationName, mockString)
      .put(GameScoringDriver.modelInputDirectory, mockPath)
      .put(GameScoringDriver.modelId, modelId)
      .put(GameScoringDriver.logDataAndModelStats, true)
      .put(GameScoringDriver.spillScoresToDisk, true)

    GameScoringDriver.validateParams(validParamMap)
  }

  @DataProvider
  def invalidParamMaps(): Array[Array[Any]] = {

    val featureShardId = "id"

    val mockPath = mock(classOf[Path])
    val mockFeatureShardConfig = mock(classOf[FeatureShardConfiguration])

    doReturn(false).when(mockFeatureShardConfig).hasIntercept

    val validParamMap = ParamMap
      .empty
      .put(GameScoringDriver.inputDataDirectories, Set[Path](mockPath))
      .put(GameScoringDriver.rootOutputDirectory, mockPath)
      .put(GameScoringDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))
      .put(GameScoringDriver.modelInputDirectory, mockPath)

    Array(
      // No input data directories
      Array(validParamMap.copy.remove(GameScoringDriver.inputDataDirectories)),
      // No root output directory
      Array(validParamMap.copy.remove(GameScoringDriver.rootOutputDirectory)),
      // No feature bags directory
      Array(validParamMap.copy.remove(GameScoringDriver.featureBagsDirectory)),
      // No feature shard configurations
      Array(validParamMap.copy.remove(GameScoringDriver.featureShardConfigurations)),
      // Off-heap map dir without partitions
      Array(validParamMap.copy.put(GameScoringDriver.offHeapIndexMapDirectory, mockPath)),
      // Off-heap map partitions without dir
      Array(validParamMap.copy.put(GameScoringDriver.offHeapIndexMapPartitions, 1)),
      // Both off-heap map and features directory
      Array(
        validParamMap
          .copy
          .put(GameScoringDriver.offHeapIndexMapDirectory, mockPath)
          .put(GameScoringDriver.featureBagsDirectory, mockPath)),
      // No model input directory
      Array(validParamMap.copy.remove(GameScoringDriver.modelInputDirectory)))
  }

  /**
   * Test that invalid parameters will be correctly rejected.
   *
   * @param params A [[ParamMap]] with one or more flaws
   */
  @Test(dataProvider = "invalidParamMaps", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testValidateParams(params: ParamMap): Unit = {

    GameScoringDriver.clear()

    params.toSeq.foreach(pair => GameScoringDriver.set(pair.param.asInstanceOf[Param[Any]], pair.value))
    GameScoringDriver.validateParams()
  }

  /**
   * Test that default values are set for all parameters that require them.
   */
  @Test
  def testDefaultParams(): Unit = {

    GameScoringDriver.clear()

    GameScoringDriver.getOrDefault(GameScoringDriver.inputColumnNames)
    GameScoringDriver.getOrDefault(GameScoringDriver.overrideOutputDirectory)
    GameScoringDriver.getOrDefault(GameScoringDriver.logDataAndModelStats)
    GameScoringDriver.getOrDefault(GameScoringDriver.spillScoresToDisk)
    GameScoringDriver.getOrDefault(GameScoringDriver.dataValidation)
    GameScoringDriver.getOrDefault(GameScoringDriver.logLevel)
    GameScoringDriver.getOrDefault(GameScoringDriver.applicationName)
  }

  /**
   * Test that set parameters can be cleared correctly.
   */
  @Test
  def testClear(): Unit = {

    val mockPath = mock(classOf[Path])

    GameScoringDriver.set(GameScoringDriver.rootOutputDirectory, mockPath)

    assertEquals(GameScoringDriver.get(GameScoringDriver.rootOutputDirectory), Some(mockPath))

    GameScoringDriver.clear()

    assertEquals(GameScoringDriver.get(GameScoringDriver.rootOutputDirectory), None)
  }
}
