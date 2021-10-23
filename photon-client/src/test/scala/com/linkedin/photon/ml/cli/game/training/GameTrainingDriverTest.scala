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
package com.linkedin.photon.ml.cli.game.training

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.{DataValidationType, HyperparameterTunerName, HyperparameterTuningMode, TaskType}
import com.linkedin.photon.ml.data.{CoordinateDataConfiguration, InputColumnsNames, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.estimators.GameEstimator
import com.linkedin.photon.ml.evaluation.{EvaluationResults, EvaluatorType}
import com.linkedin.photon.ml.io.{CoordinateConfiguration, FeatureShardConfiguration, ModelOutputMode, RandomEffectCoordinateConfiguration}
import com.linkedin.photon.ml.io.ModelOutputMode.ModelOutputMode
import com.linkedin.photon.ml.model.GameModel
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.VarianceComputationType
import com.linkedin.photon.ml.util.{DateRange, PhotonLogger}

/**
 * Unit tests for [[GameTrainingDriver]].
 */
class GameTrainingDriverTest {

  /**
   * Test that a [[ParamMap]] with only required parameters set can be valid input.
   */
  @Test
  def testMinValidParamMap(): Unit = {

    val coordinateId = "id"
    val featureShardId = "id"

    val mockPath = mock(classOf[Path])
    val mockFeatureShardConfig = mock(classOf[FeatureShardConfiguration])
    val mockDataConfig = mock(classOf[CoordinateDataConfiguration])
    val mockCoordinateConfig = mock(classOf[CoordinateConfiguration])

    doReturn(false).when(mockFeatureShardConfig).hasIntercept
    doReturn(mockDataConfig).when(mockCoordinateConfig).dataConfiguration
    doReturn(featureShardId).when(mockDataConfig).featureShardId

    val validParamMap = ParamMap
      .empty
      .put(GameTrainingDriver.inputDataDirectories, Set[Path](mockPath))
      .put(GameTrainingDriver.rootOutputDirectory, mockPath)
      .put(GameTrainingDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))
      .put(GameTrainingDriver.trainingTask, TaskType.LINEAR_REGRESSION)
      .put(GameTrainingDriver.coordinateDescentIterations, 1)
      .put(GameTrainingDriver.coordinateConfigurations, Map((coordinateId, mockCoordinateConfig)))
      .put(GameTrainingDriver.coordinateUpdateSequence, Seq(coordinateId))

    GameTrainingDriver.validateParams(validParamMap)
  }

  /**
   * Test that a [[ParamMap]] with all parameters set can be valid input.
   */
  @Test
  def testMaxValidParamMap(): Unit = {

    val coordinateId1 = "id1"
    val coordinateId2 = "id2"
    val featureShardId = "id"

    val mockString = "text"
    val mockInt = 10
    val mockVarianceComputationType = VarianceComputationType.FULL
    val mockBoolean = true

    val mockPath = mock(classOf[Path])
    val mockDateRange = mock(classOf[DateRange])
    val mockInputColumnNames = mock(classOf[InputColumnsNames])
    val mockEvaluatorType = mock(classOf[EvaluatorType])
    val mockFeatureShardConfig = mock(classOf[FeatureShardConfiguration])
    val mockDataConfig = mock(classOf[CoordinateDataConfiguration])
    val mockCoordinateConfig = mock(classOf[CoordinateConfiguration])

    doReturn(true).when(mockFeatureShardConfig).hasIntercept
    doReturn(mockDataConfig).when(mockCoordinateConfig).dataConfiguration
    doReturn(featureShardId).when(mockDataConfig).featureShardId

    val validParamMap = ParamMap
      .empty
      .put(GameTrainingDriver.inputDataDirectories, Set[Path](mockPath))
      .put(GameTrainingDriver.inputDataDateRange, mockDateRange)
      .put(GameTrainingDriver.offHeapIndexMapDirectory, mockPath)
      .put(GameTrainingDriver.offHeapIndexMapPartitions, mockInt)
      .put(GameTrainingDriver.inputColumnNames, mockInputColumnNames)
      .put(GameTrainingDriver.evaluators, Seq[EvaluatorType](mockEvaluatorType))
      .put(GameTrainingDriver.rootOutputDirectory, mockPath)
      .put(GameTrainingDriver.overrideOutputDirectory, mockBoolean)
      .put(GameTrainingDriver.outputFilesLimit, mockInt)
      .put(GameTrainingDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))
      .put(GameTrainingDriver.dataValidation, DataValidationType.VALIDATE_FULL)
      .put(GameTrainingDriver.logLevel, mockInt)
      .put(GameTrainingDriver.applicationName, mockString)
      .put(GameTrainingDriver.trainingTask, TaskType.LINEAR_REGRESSION)
      .put(GameTrainingDriver.validationDataDirectories, Set[Path](mockPath))
      .put(GameTrainingDriver.validationDataDateRange, mockDateRange)
      .put(GameTrainingDriver.minValidationPartitions, mockInt)
      .put(GameTrainingDriver.modelInputDirectory, mockPath)
      .put(GameTrainingDriver.partialRetrainLockedCoordinates, Set(coordinateId2))
      .put(GameTrainingDriver.outputMode, ModelOutputMode.TUNED)
      .put(GameTrainingDriver.coordinateDescentIterations, mockInt)
      .put(GameTrainingDriver.coordinateConfigurations, Map((coordinateId1, mockCoordinateConfig)))
      .put(GameTrainingDriver.coordinateUpdateSequence, Seq(coordinateId1, coordinateId2))
      .put(GameTrainingDriver.normalization, NormalizationType.STANDARDIZATION)
      .put(GameTrainingDriver.dataSummaryDirectory, mockPath)
      .put(GameTrainingDriver.treeAggregateDepth, mockInt)
      .put(GameTrainingDriver.hyperParameterTunerName, HyperparameterTunerName.DUMMY)
      .put(GameTrainingDriver.hyperParameterTuning, HyperparameterTuningMode.BAYESIAN)
      .put(GameTrainingDriver.hyperParameterTuningIter, mockInt)
      .put(GameTrainingDriver.varianceComputationType, mockVarianceComputationType)

    GameTrainingDriver.validateParams(validParamMap)
  }

  @DataProvider
  def invalidParamMaps(): Array[Array[Any]] = {

    val mockPath = mock(classOf[Path])
    val mockFeatureShardConfig = mock(classOf[FeatureShardConfiguration])
    val mockDataConfig = mock(classOf[CoordinateDataConfiguration])
    val mockCoordinateConfig = mock(classOf[CoordinateConfiguration])
    val mockBadDataConfig = mock(classOf[CoordinateDataConfiguration])
    val mockBadCoordinateConfig = mock(classOf[CoordinateConfiguration])
    val mockRECoordinateConfig = mock(classOf[RandomEffectCoordinateConfiguration])
    val mockREDataConfig = mock(classOf[RandomEffectDataConfiguration])

    val coordinateId1 = "id1"
    val coordinateId2 = "id2"
    val featureShardId = "id"
    val missingCoordinateId = "missing"
    val missingFeatureShardId = "missing"
    val badREType = "uid"
    val updateSequence1 = Seq(coordinateId1, coordinateId2)
    val updateSequence2 = Seq(coordinateId1)
    val updateSequence3 = Seq(coordinateId2)
    val coordinateConfigs1 = Map(coordinateId1 -> mockCoordinateConfig, coordinateId2 -> mockCoordinateConfig)
    val coordinateConfigs2 = Map(coordinateId2 -> mockCoordinateConfig)
    val lockedCoordinates = Set(coordinateId1)

    doReturn(false).when(mockFeatureShardConfig).hasIntercept
    doReturn(mockDataConfig).when(mockCoordinateConfig).dataConfiguration
    doReturn(featureShardId).when(mockDataConfig).featureShardId
    doReturn(mockBadDataConfig).when(mockBadCoordinateConfig).dataConfiguration
    doReturn(missingFeatureShardId).when(mockBadDataConfig).featureShardId
    doReturn(mockREDataConfig).when(mockRECoordinateConfig).dataConfiguration
    doReturn(featureShardId).when(mockREDataConfig).featureShardId
    doReturn(badREType).when(mockREDataConfig).randomEffectType

    val validParamMap = ParamMap
      .empty
      .put(GameTrainingDriver.inputDataDirectories, Set[Path](mockPath))
      .put(GameTrainingDriver.rootOutputDirectory, mockPath)
      .put(GameTrainingDriver.featureShardConfigurations, Map((featureShardId, mockFeatureShardConfig)))
      .put(GameTrainingDriver.trainingTask, TaskType.LINEAR_REGRESSION)
      .put(GameTrainingDriver.coordinateDescentIterations, 1)
      .put(GameTrainingDriver.coordinateConfigurations, coordinateConfigs1)
      .put(GameTrainingDriver.coordinateUpdateSequence, updateSequence1)

    Array(
      // No input data directories
      Array(validParamMap.copy.remove(GameTrainingDriver.inputDataDirectories)),
      // No root output directory
      Array(validParamMap.copy.remove(GameTrainingDriver.rootOutputDirectory)),
      // No feature shard configurations
      Array(validParamMap.copy.remove(GameTrainingDriver.featureShardConfigurations)),
      // Off-heap map dir without partitions
      Array(validParamMap.copy.put(GameTrainingDriver.offHeapIndexMapDirectory, mockPath)),
      // Off-heap map partitions without dir
      Array(validParamMap.copy.put(GameTrainingDriver.offHeapIndexMapPartitions, 1)),
      // Both off-heap map and features directory
      Array(
        validParamMap
          .copy
          .put(GameTrainingDriver.offHeapIndexMapDirectory, mockPath)
          .put(GameTrainingDriver.featureBagsDirectory, mockPath)),
      // No training task
      Array(validParamMap.copy.remove(GameTrainingDriver.trainingTask)),
      // No coordinate descent iterations
      Array(validParamMap.copy.remove(GameTrainingDriver.coordinateDescentIterations)),
      // No coordinate configurations
      Array(validParamMap.copy.remove(GameTrainingDriver.coordinateConfigurations)),
      // No coordinate update sequence
      Array(validParamMap.copy.remove(GameTrainingDriver.coordinateUpdateSequence)),
      // Locked coordinates without pre-trained model path
      Array(validParamMap.copy.put(GameTrainingDriver.partialRetrainLockedCoordinates, lockedCoordinates)),
      // Locked coordinate present in the coordinate configurations
      Array(
        validParamMap
          .copy
          .put(GameTrainingDriver.modelInputDirectory, mockPath)
          .put(GameTrainingDriver.partialRetrainLockedCoordinates, lockedCoordinates)),
      // All coordinates in the update sequence are locked
      Array(
        validParamMap
          .copy
          .put(GameTrainingDriver.coordinateUpdateSequence, updateSequence2)
          .put(GameTrainingDriver.coordinateConfigurations, coordinateConfigs2)
          .put(GameTrainingDriver.modelInputDirectory, mockPath)
          .put(GameTrainingDriver.partialRetrainLockedCoordinates, lockedCoordinates)),
      // Locked coordinate missing from the update sequence
      Array(
        validParamMap
          .copy
          .put(GameTrainingDriver.coordinateUpdateSequence, updateSequence3)
          .put(GameTrainingDriver.coordinateConfigurations, coordinateConfigs2)
          .put(GameTrainingDriver.modelInputDirectory, mockPath)
          .put(GameTrainingDriver.partialRetrainLockedCoordinates, lockedCoordinates)),
      // Missing coordinate configuration
      Array(validParamMap.copy.put(GameTrainingDriver.coordinateUpdateSequence, Seq(missingCoordinateId))),
      // Missing feature shard configuration
      Array(
        validParamMap
          .copy
          .put(GameTrainingDriver.coordinateConfigurations, Map((coordinateId1, mockBadCoordinateConfig)))),
      // Random effect coordinate configuration with invalid random effect field
      Array(
        validParamMap
          .copy
          .put(GameTrainingDriver.coordinateConfigurations, Map((coordinateId1, mockRECoordinateConfig)))),
      // No intercepts for standardization
      Array(validParamMap.copy.put(GameTrainingDriver.normalization, NormalizationType.STANDARDIZATION)),
      // No iterations for hyperparameter tuning
      Array(validParamMap.copy.put(GameTrainingDriver.hyperParameterTuning, HyperparameterTuningMode.BAYESIAN)))
  }

  /**
   * Test that invalid parameters will be correctly rejected.
   *
   * @param params A [[ParamMap]] with one or more flaws
   */
  @Test(dataProvider = "invalidParamMaps", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testValidateParams(params: ParamMap): Unit = GameTrainingDriver.validateParams(params)

  /**
   * Test that default values are set for all parameters that require them.
   */
  @Test
  def testDefaultParams(): Unit = {

    GameTrainingDriver.clear()

    GameTrainingDriver.getOrDefault(GameTrainingDriver.inputColumnNames)
    GameTrainingDriver.getOrDefault(GameTrainingDriver.outputMode)
    GameTrainingDriver.getOrDefault(GameTrainingDriver.overrideOutputDirectory)
    GameTrainingDriver.getOrDefault(GameTrainingDriver.normalization)
    GameTrainingDriver.getOrDefault(GameTrainingDriver.hyperParameterTunerName)
    GameTrainingDriver.getOrDefault(GameTrainingDriver.hyperParameterTuning)
    GameTrainingDriver.getOrDefault(GameTrainingDriver.varianceComputationType)
    GameTrainingDriver.getOrDefault(GameTrainingDriver.dataValidation)
    GameTrainingDriver.getOrDefault(GameTrainingDriver.logLevel)
    GameTrainingDriver.getOrDefault(GameTrainingDriver.applicationName)
  }

  /**
   * Test that set parameters can be cleared correctly.
   */
  @Test
  def testClear(): Unit = {

    val mockPath = mock(classOf[Path])

    GameTrainingDriver.set(GameTrainingDriver.rootOutputDirectory, mockPath)

    assertEquals(GameTrainingDriver.get(GameTrainingDriver.rootOutputDirectory), Some(mockPath))

    GameTrainingDriver.clear()

    assertEquals(GameTrainingDriver.get(GameTrainingDriver.rootOutputDirectory), None)
  }

  @DataProvider
  def modelOutputModes(): Array[Array[Any]] = {

    val mockGameModel = mock(classOf[GameModel])
    val mockGameOptConfig = mock(classOf[GameEstimator.GameOptimizationConfiguration])
    val mockEvaluatorType = mock(classOf[EvaluatorType])
    val mockEvaluationResults = mock(classOf[EvaluationResults])

    val explicitModels = Seq((mockGameModel, mockGameOptConfig, Some(mockEvaluationResults)))
    val tunedModels = Seq((mockGameModel, mockGameOptConfig, Some(mockEvaluationResults)))
    val allModels = explicitModels ++ tunedModels

    doReturn(mockEvaluatorType).when(mockEvaluationResults).primaryEvaluator
    doReturn(true).when(mockEvaluatorType).betterThan(Matchers.any(), Matchers.any())
    doReturn("").when(mockGameModel).toSummaryString
    doReturn(Seq()).when(mockGameOptConfig).toSeq

    Array(
      Array(
        ModelOutputMode.ALL,
        explicitModels,
        tunedModels,
        allModels,
        true),
      Array(
        ModelOutputMode.TUNED,
        explicitModels,
        tunedModels,
        tunedModels,
        true),
      Array(
        ModelOutputMode.EXPLICIT,
        explicitModels,
        tunedModels,
        explicitModels,
        true),
      Array(
        ModelOutputMode.BEST,
        explicitModels,
        tunedModels,
        Seq(),
        true),
      Array(
        ModelOutputMode.NONE,
        explicitModels,
        tunedModels,
        Seq(),
        false))
  }

  /**
   * Test that the correct group of trained models is selected for output to HDFS.
   *
   * @param outputMode The model output specifier
   * @param explicitModels The models trained for explicit regularization values
   * @param tunedModels The models trained during hyperparameter tuning
   * @param resultOutputModels The expected models to output to HDFS
   * @param resultBestModel Whether the best model has been selected
   */
  @Test(dataProvider = "modelOutputModes")
  def testSelectModels(
      outputMode: ModelOutputMode,
      explicitModels: Seq[GameEstimator.GameResult],
      tunedModels: Seq[GameEstimator.GameResult],
      resultOutputModels: Seq[GameEstimator.GameResult],
      resultBestModel: Boolean): Unit = {

    val mockLogger = mock(classOf[PhotonLogger])

    GameTrainingDriver.logger = mockLogger
    GameTrainingDriver.set(GameTrainingDriver.outputMode, outputMode)

    val (outputModels, bestModel) = GameTrainingDriver.selectModels(explicitModels, tunedModels)

    assertEquals(outputModels, resultOutputModels)
    assertEquals(bestModel.isDefined, resultBestModel)
  }
}
