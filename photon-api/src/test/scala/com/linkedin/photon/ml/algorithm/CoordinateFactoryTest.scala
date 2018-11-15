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
package com.linkedin.photon.ml.algorithm

import org.apache.spark.rdd.RDD
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.annotations.Test

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.data.{FixedEffectDataset, LocalDataset, RandomEffectDataset, RandomEffectDatasetInProjectedSpace}
import com.linkedin.photon.ml.function.{DistributedObjectiveFunction, ObjectiveFunctionHelper, SingleNodeObjectiveFunction}
import com.linkedin.photon.ml.normalization.NormalizationContextBroadcast
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.optimization.{OptimizerConfig, OptimizerType, VarianceComputationType}
import com.linkedin.photon.ml.sampling.DownSamplerHelper
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel

/**
 * Unit tests for the [[CoordinateFactory]].
 */
class CoordinateFactoryTest {

  import CoordinateFactoryTest._

  /**
   * Test that the [[CoordinateFactory]] can correctly build a [[FixedEffectCoordinate]].
   */
  @Test
  def testBuildFixedEffectCoordinate(): Unit = {

    val mockDataset = mock(classOf[FixedEffectDataset])
    val optimizationConfiguration = FixedEffectOptimizationConfiguration(OPTIMIZER_CONFIG)

    val coordinate = CoordinateFactory.build(
      mockDataset,
      optimizationConfiguration,
      LOSS_FUNCTION_FACTORY,
      GLM_CONSTRUCTOR,
      DOWN_SAMPLER_FACTORY,
      MOCK_NORMALIZATION_BROADCAST,
      VARIANCE_COMPUTATION_TYPE,
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
  def testBuildRandomEffectCoordinate(): Unit = {

    val mockDataset: RandomEffectDataset = mock(classOf[RandomEffectDatasetInProjectedSpace])
    val mockRDD = mock(classOf[RDD[(REId, LocalDataset)]])
    val optimizationConfiguration = RandomEffectOptimizationConfiguration(OPTIMIZER_CONFIG)

    doReturn(mockRDD).when(mockDataset).activeData
    doReturn(mockRDD).when(mockRDD).mapValues(Matchers.any(classOf[Function1[LocalDataset, SingleNodeObjectiveFunction]]))

    val coordinate = CoordinateFactory.build(
      mockDataset,
      optimizationConfiguration,
      LOSS_FUNCTION_FACTORY,
      GLM_CONSTRUCTOR,
      DOWN_SAMPLER_FACTORY,
      MOCK_NORMALIZATION_BROADCAST,
      VARIANCE_COMPUTATION_TYPE,
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
  def testBuildInvalidCoordinate(): Unit = {

    val mockDataset = mock(classOf[FixedEffectDataset])
    val optimizationConfiguration = RandomEffectOptimizationConfiguration(OPTIMIZER_CONFIG)

    CoordinateFactory.build(
      mockDataset,
      optimizationConfiguration,
      LOSS_FUNCTION_FACTORY,
      GLM_CONSTRUCTOR,
      DOWN_SAMPLER_FACTORY,
      MOCK_NORMALIZATION_BROADCAST,
      VARIANCE_COMPUTATION_TYPE,
      TRACK_STATE)
  }
}

object CoordinateFactoryTest {

  private val TRAINING_TASK = TaskType.LOGISTIC_REGRESSION
  private val OPTIMIZER_TYPE = OptimizerType.LBFGS
  private val MAX_ITER = 1
  private val TOLERANCE = 2E-2
  private val TREE_AGGREGATE_DEPTH = 1
  private val VARIANCE_COMPUTATION_TYPE = VarianceComputationType.NONE
  private val TRACK_STATE = true

  private val OPTIMIZER_CONFIG = OptimizerConfig(OPTIMIZER_TYPE, MAX_ITER, TOLERANCE)
  private val MOCK_NORMALIZATION_BROADCAST = mock(classOf[NormalizationContextBroadcast])
  private val GLM_CONSTRUCTOR = LogisticRegressionModel.apply _
  private val LOSS_FUNCTION_FACTORY = ObjectiveFunctionHelper.buildFactory(TRAINING_TASK, TREE_AGGREGATE_DEPTH)
  private val DOWN_SAMPLER_FACTORY = DownSamplerHelper.buildFactory(TRAINING_TASK)
}
