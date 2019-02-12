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

import breeze.linalg.DenseVector
import org.mockito.Mockito._
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NoNormalization
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.RandomEffectOptimizationConfiguration
import com.linkedin.photon.ml.util.PhotonNonBroadcast

/**
 * Unit tests for [[SingleNodeSmoothedHingeLossFunction]].
 */
class SingleNodeSmoothedHingeLossFunctionTest {

  import SingleNodeSmoothedHingeLossFunctionTest._

  /**
   * Verify the value of loss function without regularization.
   */
  @Test()
  def testValueNoRegularization(): Unit = {

    val labeledPoints = Iterable(LABELED_POINT_1, LABELED_POINT_2)
    val coefficients = COEFFICIENT_VECTOR

    val randomEffectRegularizationContext = NoRegularizationContext
    val randomEffectOptimizationConfiguration = RandomEffectOptimizationConfiguration(
      RANDOM_EFFECT_OPTIMIZER_CONFIG,
      randomEffectRegularizationContext)
    val singleNodeSmoothedHingeLossFunction = SingleNodeSmoothedHingeLossFunction(randomEffectOptimizationConfiguration)
    val value = singleNodeSmoothedHingeLossFunction.value(
      labeledPoints,
      coefficients,
      PhotonNonBroadcast(NORMALIZATION_CONTEXT))

    assertEquals(value, 6.0, EPSILON)
  }

  /**
   * Verify the value of loss function with L2 regularization.
   */
  @Test()
  def testValueWithL2Regularization(): Unit = {

    val labeledPoints = Iterable(LABELED_POINT_1, LABELED_POINT_2)
    val coefficients = COEFFICIENT_VECTOR

    val randomEffectRegularizationContext = L2RegularizationContext
    val randomEffectOptimizationConfiguration = RandomEffectOptimizationConfiguration(
      RANDOM_EFFECT_OPTIMIZER_CONFIG,
      randomEffectRegularizationContext,
      RANDOM_EFFECT_REGULARIZATION_WEIGHT)
    val singleNodeSmoothedHingeLossFunction = SingleNodeSmoothedHingeLossFunction(randomEffectOptimizationConfiguration)
    val value = singleNodeSmoothedHingeLossFunction.value(
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
  def testValueWithElasticNetRegularization(): Unit = {

    val labeledPoints = Iterable(LABELED_POINT_1, LABELED_POINT_2)
    val coefficients = COEFFICIENT_VECTOR

    val randomEffectRegularizationContext = ElasticNetRegularizationContext(ALPHA)
    val randomEffectOptimizationConfiguration = RandomEffectOptimizationConfiguration(
      RANDOM_EFFECT_OPTIMIZER_CONFIG,
      randomEffectRegularizationContext,
      RANDOM_EFFECT_REGULARIZATION_WEIGHT)
    val singleNodeSmoothedHingeLossFunction = SingleNodeSmoothedHingeLossFunction(randomEffectOptimizationConfiguration)
    val value = singleNodeSmoothedHingeLossFunction.value(
      labeledPoints,
      coefficients,
      PhotonNonBroadcast(NORMALIZATION_CONTEXT))

    // L1 is computed by the optimizer.
    // expectedValue = 6 + (1 - 0.4) * 1 * ((-2)^2 + 3^2) / 2 = 9.9
    assertEquals(value, 9.9, EPSILON)
  }
}

object SingleNodeSmoothedHingeLossFunctionTest {

  private val RANDOM_EFFECT_OPTIMIZER_CONFIG = mock(classOf[OptimizerConfig])
  private val LABELED_POINT_1 = new LabeledPoint(0, DenseVector(0.0, 1.0))
  private val LABELED_POINT_2 = new LabeledPoint(1, DenseVector(1.0, 0.0))
  private val COEFFICIENT_VECTOR = DenseVector(-2.0, 3.0)
  private val NORMALIZATION_CONTEXT = NoNormalization()
  private val RANDOM_EFFECT_REGULARIZATION_WEIGHT = 1D
  private val ALPHA = 0.4
  private val EPSILON = 1e-3
}
