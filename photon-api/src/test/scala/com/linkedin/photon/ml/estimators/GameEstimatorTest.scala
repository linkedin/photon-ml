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
package com.linkedin.photon.ml.estimators

import org.apache.commons.cli.MissingArgumentException
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.DataFrame
import org.mockito.Mockito._
import org.slf4j.Logger
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.data.{CoordinateDataConfiguration, InputColumnsNames}
import com.linkedin.photon.ml.evaluation.EvaluatorType.AUC
import com.linkedin.photon.ml.model.{DatumScoringModel, GameModel}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.CoordinateOptimizationConfiguration

/**
 * Unit tests for [[GameEstimator]].
 */
class GameEstimatorTest {

  /**
   * Test that a [[ParamMap]] with only required parameters set can be valid input.
   */
  @Test
  def testMinValidParamMap(): Unit = {

    val coordinateId = "id"
    val featureShardId = "id"
    val trainingTask = TaskType.LINEAR_REGRESSION

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])
    val mockDataConfig = mock(classOf[CoordinateDataConfiguration])

    doReturn(featureShardId).when(mockDataConfig).featureShardId

    val estimator = new GameEstimator(mockSparkContext, mockLogger)
      .setTrainingTask(trainingTask)
      .setCoordinateUpdateSequence(Seq(coordinateId))
      .setCoordinateDataConfigurations(Map((coordinateId, mockDataConfig)))

    estimator.validateParams()
  }

  /**
   * Test that a [[ParamMap]] with all parameters set can be valid input.
   */
  @Test
  def testMaxValidParamMap(): Unit = {

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])
    val mockInputColumnNames = mock(classOf[InputColumnsNames])
    val mockDataConfig = mock(classOf[CoordinateDataConfiguration])
    val mockNormalizationContext = mock(classOf[NormalizationContext])
    val mockDatumScoringModel = mock(classOf[DatumScoringModel])
    val mockPretrainedModel = mock(classOf[GameModel])

    val coordinateId1 = "id1"
    val coordinateId2 = "id2"
    val featureShardId = "id"
    val trainingTask = TaskType.LINEAR_REGRESSION
    val coordinateDescentIter = 1
    val computeVariance = true
    val treeAggregateDepth = 2
    val validationEvaluators = Seq(AUC)
    val updateSeq = Seq(coordinateId1, coordinateId2)
    val dataConfigs = Map((coordinateId1, mockDataConfig), (coordinateId2, mockDataConfig))
    val normalizationConfigs = Map((coordinateId1, mockNormalizationContext))
    val lockedCoordinates = Set(coordinateId2)
    val preTrainedModelMap = Map(coordinateId2 -> mockDatumScoringModel)

    doReturn(featureShardId).when(mockDataConfig).featureShardId
    doReturn(preTrainedModelMap).when(mockPretrainedModel).toMap

    val estimator = new GameEstimator(mockSparkContext, mockLogger)
      .setTrainingTask(trainingTask)
      .setInputColumnNames(mockInputColumnNames)
      .setCoordinateUpdateSequence(updateSeq)
      .setCoordinateDataConfigurations(dataConfigs)
      .setCoordinateDescentIterations(coordinateDescentIter)
      .setCoordinateNormalizationContexts(normalizationConfigs)
      .setInitialModel(mockPretrainedModel)
      .setPartialRetrainLockedCoordinates(lockedCoordinates)
      .setComputeVariance(computeVariance)
      .setTreeAggregateDepth(treeAggregateDepth)
      .setValidationEvaluators(validationEvaluators)

    estimator.validateParams()
  }

  @DataProvider
  def invalidParamMaps(): Array[Array[Any]] = {

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])
    val mockDataConfig = mock(classOf[CoordinateDataConfiguration])
    val mockNormalizationContext = mock(classOf[NormalizationContext])
    val mockDatumScoringModel = mock(classOf[DatumScoringModel])
    val mockPretrainedModel1 = mock(classOf[GameModel])
    val mockPretrainedModel2 = mock(classOf[GameModel])

    val coordinateId1 = "id1"
    val coordinateId2 = "id2"
    val coordinateId3 = "id3"
    val trainingTask = TaskType.LINEAR_REGRESSION
    val updateSeq1 = Seq(coordinateId1)
    val updateSeq2 = Seq(coordinateId1, coordinateId2)
    val updateSeq3 = Seq(coordinateId1, coordinateId2, coordinateId3)
    val badUpdateSeq = Seq(coordinateId1, coordinateId1, coordinateId1)
    val coordinateDataConfig1 = Map(coordinateId1 -> mockDataConfig)
    val coordinateDataConfig2 = Map(coordinateId1 -> mockDataConfig, coordinateId2 -> mockDataConfig)
    val lockedCoordinates1 = Set(coordinateId1)
    val lockedCoordinates2 = Set(coordinateId2)
    val preTrainedModelMap1 = Map(coordinateId1 -> mockDatumScoringModel)
    val preTrainedModelMap2 = Map(coordinateId2 -> mockDatumScoringModel)

    doReturn(preTrainedModelMap1).when(mockPretrainedModel1).toMap
    doReturn(preTrainedModelMap2).when(mockPretrainedModel2).toMap

    val estimator = new GameEstimator(mockSparkContext, mockLogger)
      .setTrainingTask(trainingTask)
      .setCoordinateUpdateSequence(updateSeq1)
      .setCoordinateDataConfigurations(coordinateDataConfig1)

    estimator.validateParams()

    var result = Seq[Array[Any]]()
    var badEstimator = estimator

    // No training task
    badEstimator = estimator.copy(ParamMap.empty)
    badEstimator.clear(badEstimator.trainingTask)
    result = result :+ Array[Any](badEstimator)

    // No coordinate update sequence
    badEstimator = estimator.copy(ParamMap.empty)
    badEstimator.clear(badEstimator.coordinateUpdateSequence)
    result = result :+ Array[Any](badEstimator)

    // No data configurations
    badEstimator = estimator.copy(ParamMap.empty)
    badEstimator.clear(badEstimator.coordinateDataConfigurations)
    result = result :+ Array[Any](badEstimator)

    // Update sequence repeats defined coordinate ID
    badEstimator = estimator
      .copy(ParamMap.empty)
      .setCoordinateUpdateSequence(badUpdateSeq)
    result = result :+ Array[Any](badEstimator)

    // Locked coordinates without pre-trained model
    badEstimator = estimator
      .copy(ParamMap.empty)
      .setPartialRetrainLockedCoordinates(lockedCoordinates1)
    result = result :+ Array[Any](badEstimator)

    // All coordinates in the update sequence are locked
    badEstimator = estimator
      .copy(ParamMap.empty)
      .setInitialModel(mockPretrainedModel1)
      .setPartialRetrainLockedCoordinates(lockedCoordinates1)
    result = result :+ Array[Any](badEstimator)

    // Locked coordinate missing from the update sequence
    badEstimator = estimator
      .copy(ParamMap.empty)
      .setInitialModel(mockPretrainedModel2)
      .setPartialRetrainLockedCoordinates(lockedCoordinates2)
    result = result :+ Array[Any](badEstimator)

    // Locked coordinate missing from the pre-trained model
    badEstimator = estimator
      .copy(ParamMap.empty)
      .setCoordinateUpdateSequence(updateSeq2)
      .setCoordinateDataConfigurations(coordinateDataConfig2)
      .setInitialModel(mockPretrainedModel2)
      .setPartialRetrainLockedCoordinates(lockedCoordinates1)
    result = result :+ Array[Any](badEstimator)

    // Update sequence has undefined coordinate ID
    badEstimator = estimator
      .copy(ParamMap.empty)
      .setCoordinateUpdateSequence(updateSeq3)
    result = result :+ Array[Any](badEstimator)

    // Normalization context undefined for coordinate ID in update sequence
    badEstimator = estimator
      .copy(ParamMap.empty)
      .setCoordinateNormalizationContexts(Map((coordinateId2, mockNormalizationContext)))
    result = result :+ Array[Any](badEstimator)

    result.toArray
  }

  /**
   * Test that invalid parameters will be correctly rejected.
   *
   * @param estimator A [[GameEstimator]] with one or more flaws in its parameters
   */
  @Test(
    dataProvider = "invalidParamMaps",
    expectedExceptions = Array(classOf[IllegalArgumentException], classOf[MissingArgumentException]))
  def testValidateParams(estimator: GameEstimator): Unit = estimator.validateParams()

  /**
   * Test that default values are set for all parameters that require them.
   */
  @Test
  def testDefaultParams(): Unit = {

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])

    val estimator = new GameEstimator(mockSparkContext, mockLogger)

    estimator.getOrDefault(estimator.coordinateDescentIterations)
    estimator.getOrDefault(estimator.inputColumnNames)
    estimator.getOrDefault(estimator.computeVariance)
    estimator.getOrDefault(estimator.treeAggregateDepth)
  }

  /**
   * Test that the [[GameEstimator]] will reject optimization configurations with locked coordinates.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidFit(): Unit = {

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])
    val mockDataConfig = mock(classOf[CoordinateDataConfiguration])
    val mockOptConfig = mock(classOf[CoordinateOptimizationConfiguration])
    val mockDatumScoringModel = mock(classOf[DatumScoringModel])
    val mockPretrainedModel = mock(classOf[GameModel])
    val mockDataFrame = mock(classOf[DataFrame])

    val coordinateId1 = "id1"
    val coordinateId2 = "id2"
    val trainingTask = TaskType.LINEAR_REGRESSION
    val updateSeq = Seq(coordinateId1, coordinateId2)
    val coordinateDataConfiguration = Map(coordinateId2 -> mockDataConfig)
    val coordinateOptConfiguration = Map(coordinateId1 -> mockOptConfig, coordinateId2 -> mockOptConfig)
    val lockedCoordinates = Set(coordinateId1)
    val preTrainedModelMap = Map(coordinateId1 -> mockDatumScoringModel)

    doReturn(preTrainedModelMap).when(mockPretrainedModel).toMap

    val estimator = new GameEstimator(mockSparkContext, mockLogger)
      .setTrainingTask(trainingTask)
      .setCoordinateUpdateSequence(updateSeq)
      .setCoordinateDataConfigurations(coordinateDataConfiguration)
      .setInitialModel(mockPretrainedModel)
      .setPartialRetrainLockedCoordinates(lockedCoordinates)

    estimator.fit(mockDataFrame, validationData = None, Seq(coordinateOptConfiguration))
  }
}
