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
package com.linkedin.photon.ml.function.glm

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
 * Integration tests for [[DistributedGLMLossFunction]].
 */
class DistributedGLMLossFunctionIntegTest extends SparkTestUtils {

  import DistributedGLMLossFunctionIntegTest._

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
    val distributedGLMLossFunction = DistributedGLMLossFunction(
      fixedEffectOptimizationConfiguration,
      LogisticLossFunction,
      TREE_AGGREGATE_DEPTH)
    val value = distributedGLMLossFunction.value(
      labeledPoints,
      coefficients,
      PhotonNonBroadcast(NORMALIZATION_CONTEXT))

    // expectValue = log(1 + exp(3)) + log(1 + exp(2)) = 5.1755
    assertEquals(value, 5.1755, EPSILON)
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
    val distributedGLMLossFunction = DistributedGLMLossFunction(
      fixedEffectOptimizationConfiguration,
      LogisticLossFunction,
      TREE_AGGREGATE_DEPTH)
    val value = distributedGLMLossFunction.value(
      labeledPoints,
      coefficients,
      PhotonNonBroadcast(NORMALIZATION_CONTEXT))

    // expectedValue = log(1 + exp(3)) + log(1 + exp(2)) + 1 * ((-2)^2 + 3^2) / 2 = 11.6755
    assertEquals(value, 11.6755, EPSILON)
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
    val distributedGLMLossFunction = DistributedGLMLossFunction(
      fixedEffectOptimizationConfiguration,
      LogisticLossFunction,
      TREE_AGGREGATE_DEPTH)
    val value = distributedGLMLossFunction.value(
      labeledPoints,
      coefficients,
      PhotonNonBroadcast(NORMALIZATION_CONTEXT))

    // L1 is computed by the optimizer.
    // expectedValue = log(1 + exp(3)) + log(1 + exp(2)) + (1 - 0.4) * 1 * ((-2)^2 + 3^2) / 2 = 9.0755
    assertEquals(value, 9.0755, EPSILON)
  }
}

object DistributedGLMLossFunctionIntegTest {

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
