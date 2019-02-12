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
package com.linkedin.photon.ml.function.svm

import breeze.linalg.{DenseVector, Vector}
import org.mockito.Mockito._
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NoNormalization
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.FixedEffectOptimizationConfiguration
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.PhotonNonBroadcast

/**
 * Integration tests for [[DistributedSmoothedHingeLossFunction]].
 */
class DistributedSmoothedHingeLossFunctionIntegTest extends SparkTestUtils {

  import DistributedSmoothedHingeLossFunctionIntegTest._

  /**
   * Verify the value of loss function without regularization.
   */
  @Test()
  def testValueNoRegularization(): Unit = sparkTest("testValueNoRegularization") {

    val labeledPoints = sc.parallelize(Array(LABELED_POINT_1, LABELED_POINT_2))
    val coefficients = sc.broadcast(COEFFICIENT_VECTOR)

    val fixedEffectRegularizationContext = NoRegularizationContext
    val fixedEffectOptimizationConfiguration = FixedEffectOptimizationConfiguration(
      FIXED_EFFECT_OPTIMIZER_CONFIG,
      fixedEffectRegularizationContext)
    val distributedSmoothedHingeLossFunction = DistributedSmoothedHingeLossFunction(
      fixedEffectOptimizationConfiguration,
      TREE_AGGREGATE_DEPTH)
    val value = distributedSmoothedHingeLossFunction.value(
      labeledPoints,
      coefficients,
      PhotonNonBroadcast(NORMALIZATION_CONTEXT))

    assertEquals(value, 6.0, EPSILON)
  }

  /**
   * Verify the value of loss function with L2 regularization.
   */
  @Test()
  def testValueWithL2Regularization(): Unit = sparkTest("testValueWithL2Regularization") {

    val labeledPoints = sc.parallelize(Array(LABELED_POINT_1, LABELED_POINT_2))
    val coefficients = sc.broadcast(COEFFICIENT_VECTOR)

    val fixedEffectRegularizationContext = L2RegularizationContext
    val fixedEffectOptimizationConfiguration = FixedEffectOptimizationConfiguration(
      FIXED_EFFECT_OPTIMIZER_CONFIG,
      fixedEffectRegularizationContext,
      FIXED_EFFECT_REGULARIZATION_WEIGHT)
    val distributedSmoothedHingeLossFunction = DistributedSmoothedHingeLossFunction(
      fixedEffectOptimizationConfiguration,
      TREE_AGGREGATE_DEPTH)
    val value = distributedSmoothedHingeLossFunction.value(
      labeledPoints,
      coefficients,
      PhotonNonBroadcast(NORMALIZATION_CONTEXT))

    // expectedValue = 6 + 1 * ((-2)^2 + 3^2) / 2 = 12.5
    assertEquals(value, 12.5, EPSILON)
  }

  /**
   * Verify the value of loss function with elastic net regularization.
   */
  @Test()
  def testValueWithElasticNetRegularization(): Unit = sparkTest("testValueWithElasticNetRegularization") {

    val labeledPoints = sc.parallelize(Array(LABELED_POINT_1, LABELED_POINT_2))
    val coefficients = sc.broadcast(COEFFICIENT_VECTOR)

    val fixedEffectRegularizationContext = ElasticNetRegularizationContext(ALPHA)
    val fixedEffectOptimizationConfiguration = FixedEffectOptimizationConfiguration(
      FIXED_EFFECT_OPTIMIZER_CONFIG,
      fixedEffectRegularizationContext,
      FIXED_EFFECT_REGULARIZATION_WEIGHT)
    val distributedSmoothedHingeLossFunction = DistributedSmoothedHingeLossFunction(
      fixedEffectOptimizationConfiguration,
      TREE_AGGREGATE_DEPTH)
    val value = distributedSmoothedHingeLossFunction.value(
      labeledPoints,
      coefficients,
      PhotonNonBroadcast(NORMALIZATION_CONTEXT))

    // L1 is computed by the optimizer.
    // expectedValue = 6 + (1 - 0.4) * 1 * ((-2)^2 + 3^2) / 2 = 9.9
    assertEquals(value, 9.9, EPSILON)
  }
}

object DistributedSmoothedHingeLossFunctionIntegTest {

  private val FIXED_EFFECT_OPTIMIZER_CONFIG = mock(classOf[OptimizerConfig])
  private val LABELED_POINT_1 = new LabeledPoint(0, DenseVector(0.0, 1.0))
  private val LABELED_POINT_2 = new LabeledPoint(1, DenseVector(1.0, 0.0))
  private val COEFFICIENT_VECTOR = Vector(-2.0, 3.0)
  private val NORMALIZATION_CONTEXT = NoNormalization()
  private val FIXED_EFFECT_REGULARIZATION_WEIGHT = 1D
  private val ALPHA = 0.4
  private val TREE_AGGREGATE_DEPTH = 2
  private val EPSILON = 1e-3
}
