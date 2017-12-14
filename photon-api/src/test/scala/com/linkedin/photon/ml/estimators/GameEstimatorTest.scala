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

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.ParamMap
import org.mockito.Mockito._
import org.slf4j.Logger
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.data.{CoordinateDataConfiguration, InputColumnsNames}
import com.linkedin.photon.ml.evaluation.EvaluatorType.AUC
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper

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

    estimator.set(estimator.trainingTask, trainingTask)
    estimator.set(estimator.coordinateUpdateSequence, Seq(coordinateId))
    estimator.set(estimator.coordinateDataConfigurations, Map((coordinateId, mockDataConfig)))

    estimator.validateParams()
  }

  /**
   * Test that a [[ParamMap]] with all parameters set can be valid input.
   */
  @Test
  def testMaxValidParamMap(): Unit = {

    val coordinateId = "id"
    val featureShardId = "id"
    val trainingTask = TaskType.LINEAR_REGRESSION
    val coordinateDescentIter = 1
    val computeVariance = true
    val treeAggregateDepth = 2
    val validationEvaluators = Seq(AUC)
    val useWarmStart = false

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])
    val mockInputColumnNames = mock(classOf[InputColumnsNames])
    val mockDataConfig = mock(classOf[CoordinateDataConfiguration])
    val mockNormalizationBroadcast = mock(classOf[BroadcastWrapper[NormalizationContext]])

    doReturn(featureShardId).when(mockDataConfig).featureShardId

    val estimator = new GameEstimator(mockSparkContext, mockLogger)

    estimator.set(estimator.trainingTask, trainingTask)
    estimator.set(estimator.inputColumnNames, mockInputColumnNames)
    estimator.set(estimator.coordinateUpdateSequence, Seq(coordinateId))
    estimator.set(estimator.coordinateDataConfigurations, Map((coordinateId, mockDataConfig)))
    estimator.set(estimator.coordinateDescentIterations, coordinateDescentIter)
    estimator.set(estimator.coordinateNormalizationContexts, Map((coordinateId, mockNormalizationBroadcast)))
    estimator.set(estimator.computeVariance, computeVariance)
    estimator.set(estimator.treeAggregateDepth, treeAggregateDepth)
    estimator.set(estimator.validationEvaluators, validationEvaluators)
    estimator.set(estimator.useWarmStart, useWarmStart)

    estimator.validateParams()
  }

  @DataProvider
  def invalidParamMaps(): Array[Array[Any]] = {

    val coordinateId1 = "id1"
    val coordinateId2 = "id2"
    val trainingTask = TaskType.LINEAR_REGRESSION

    val badUpdateSeq1 = Seq(coordinateId1, coordinateId1, coordinateId1)
    val badUpdateSeq2 = Seq(coordinateId1, coordinateId2)

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])
    val mockDataConfig1 = mock(classOf[CoordinateDataConfiguration])
    val mockNormalizationBroadcast = mock(classOf[BroadcastWrapper[NormalizationContext]])

    val estimator = new GameEstimator(mockSparkContext, mockLogger)

    estimator.set(estimator.trainingTask, trainingTask)
    estimator.set(estimator.coordinateUpdateSequence, Seq(coordinateId1))
    estimator.set(estimator.coordinateDataConfigurations, Map((coordinateId1, mockDataConfig1)))

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
    badEstimator = estimator.copy(ParamMap.empty)
    badEstimator.set(badEstimator.coordinateUpdateSequence, badUpdateSeq1)
    result = result :+ Array[Any](badEstimator)

    // Update sequence has undefined coordinate ID
    badEstimator = estimator.copy(ParamMap.empty)
    badEstimator.set(badEstimator.coordinateUpdateSequence, badUpdateSeq2)
    result = result :+ Array[Any](badEstimator)

    // Normalization context undefined for coordinate ID in update sequence
    badEstimator = estimator.copy(ParamMap.empty)
    badEstimator.set(badEstimator.coordinateNormalizationContexts, Map((coordinateId2, mockNormalizationBroadcast)))
    result = result :+ Array[Any](badEstimator)

    result.toArray
  }

  /**
   * Test that invalid parameters will be correctly rejected.
   *
   * @param estimator A [[GameEstimator]] with one or more flaws in its parameters
   */
  @Test(dataProvider = "invalidParamMaps", expectedExceptions = Array(classOf[IllegalArgumentException]))
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
    estimator.getOrDefault(estimator.useWarmStart)
  }
}
