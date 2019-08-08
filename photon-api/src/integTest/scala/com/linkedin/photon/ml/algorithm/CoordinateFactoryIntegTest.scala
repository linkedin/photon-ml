/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.algorithm

import org.apache.spark.rdd.RDD
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.annotations.Test

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.data.{FixedEffectDataset, LocalDataset, RandomEffectDataset}
import com.linkedin.photon.ml.function.{DistributedObjectiveFunction, ObjectiveFunctionHelper, SingleNodeObjectiveFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.optimization.{OptimizerConfig, OptimizerType, SingleNodeOptimizationProblem, VarianceComputationType}
import com.linkedin.photon.ml.projector.LinearSubspaceProjector
import com.linkedin.photon.ml.sampling.DownSamplerHelper
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * Unit tests for the [[CoordinateFactory]].
 */
class CoordinateFactoryIntegTest extends SparkTestUtils {

  import CoordinateFactoryIntegTest._

  /**
   * Test that the [[CoordinateFactory]] can correctly build a [[FixedEffectCoordinate]].
   */
  @Test
  def testBuildFixedEffectCoordinate(): Unit = sparkTest("testBuildFixedEffectCoordinate") {

    val mockDataset = mock(classOf[FixedEffectDataset])
    val optimizationConfiguration = FixedEffectOptimizationConfiguration(OPTIMIZER_CONFIG)

    doReturn(sc).when(mockDataset).sparkContext

    val coordinate = CoordinateFactory.build(
      mockDataset,
      optimizationConfiguration,
      LOSS_FUNCTION_FACTORY,
      GLM_CONSTRUCTOR,
      DOWN_SAMPLER_FACTORY,
      MOCK_NORMALIZATION,
      VARIANCE_COMPUTATION_TYPE,
      INTERCEPT_INDEX,
      TRACK_STATE)

    coordinate match {
      case _: FixedEffectCoordinate[DistributedObjectiveFunction] =>
      case other =>
        throw new IllegalArgumentException(
          s"Expected FixedEffectCoordinate[DistributedObjectiveFunction] but got '${other.getClass.toString}'")
    }
  }

  /**
   * Test that the [[CoordinateFactory]] can correctly build a [[RandomEffectCoordinate]].
   */
  @Test
  def testBuildRandomEffectCoordinate(): Unit = sparkTest("testBuildRandomEffectCoordinate") {

    val mockDataset: RandomEffectDataset = mock(classOf[RandomEffectDataset])
    val mockDataRDD = mock(classOf[RDD[(REId, LocalDataset)]])
    val mockProjectorsRDD = mock(classOf[RDD[(REId, LinearSubspaceProjector)]])
    val mockProblemsRDD = mock(classOf[RDD[(REId, SingleNodeOptimizationProblem[SingleNodeObjectiveFunction])]])
    val optimizationConfiguration = RandomEffectOptimizationConfiguration(OPTIMIZER_CONFIG)

    doReturn(sc).when(mockDataset).sparkContext
    doReturn(mockDataRDD).when(mockDataset).activeData
    doReturn(mockDataRDD)
      .when(mockDataRDD)
      .mapValues(Matchers.any(classOf[Function1[LocalDataset, SingleNodeObjectiveFunction]]))
    doReturn(mockProjectorsRDD).when(mockDataset).projectors
    doReturn(mockProblemsRDD)
      .when(mockProjectorsRDD)
      .mapValues(Matchers.any(classOf[Function1[LinearSubspaceProjector, SingleNodeOptimizationProblem[SingleNodeObjectiveFunction]]]))

    val coordinate = CoordinateFactory.build(
      mockDataset,
      optimizationConfiguration,
      LOSS_FUNCTION_FACTORY,
      GLM_CONSTRUCTOR,
      DOWN_SAMPLER_FACTORY,
      MOCK_NORMALIZATION,
      VARIANCE_COMPUTATION_TYPE,
      INTERCEPT_INDEX,
      TRACK_STATE)

    coordinate match {
      case _: RandomEffectCoordinate[SingleNodeObjectiveFunction] =>
      case other =>
        throw new IllegalArgumentException(
          s"Expected RandomEffectCoordinate[SingleNodeObjectiveFunction] but got '${other.getClass.toString}'")
    }
  }

  /**
   * Test that the [[CoordinateFactory]] will reject invalid combinations of inputs.
   */
  @Test(expectedExceptions = Array(classOf[UnsupportedOperationException]))
  def testBuildInvalidCoordinate(): Unit = sparkTest("testBuildInvalidCoordinate") {

    val mockDataset = mock(classOf[FixedEffectDataset])
    val optimizationConfiguration = RandomEffectOptimizationConfiguration(OPTIMIZER_CONFIG)

    CoordinateFactory.build(
      mockDataset,
      optimizationConfiguration,
      LOSS_FUNCTION_FACTORY,
      GLM_CONSTRUCTOR,
      DOWN_SAMPLER_FACTORY,
      MOCK_NORMALIZATION,
      VARIANCE_COMPUTATION_TYPE,
      INTERCEPT_INDEX,
      TRACK_STATE)
  }
}

object CoordinateFactoryIntegTest {

  private val TRAINING_TASK = TaskType.LOGISTIC_REGRESSION
  private val OPTIMIZER_TYPE = OptimizerType.LBFGS
  private val MAX_ITER = 1
  private val TOLERANCE = 2E-2
  private val TREE_AGGREGATE_DEPTH = 1
  private val VARIANCE_COMPUTATION_TYPE = VarianceComputationType.NONE
  private val TRACK_STATE = true

  private val OPTIMIZER_CONFIG = OptimizerConfig(OPTIMIZER_TYPE, MAX_ITER, TOLERANCE)
  private val MOCK_NORMALIZATION = mock(classOf[NormalizationContext])
  private val GLM_CONSTRUCTOR = LogisticRegressionModel.apply _
  private val LOSS_FUNCTION_FACTORY = ObjectiveFunctionHelper.buildFactory(TRAINING_TASK, TREE_AGGREGATE_DEPTH)
  private val DOWN_SAMPLER_FACTORY = DownSamplerHelper.buildFactory(TRAINING_TASK)
  private val INTERCEPT_INDEX = None
}
