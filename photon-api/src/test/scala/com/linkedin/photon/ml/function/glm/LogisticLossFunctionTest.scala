/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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

import breeze.linalg.DenseVector
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst

/**
 * Test some edge cases of the functions in [[LogisticLossFunction]]. For more tests by numerical methods please see
 * [[com.linkedin.photon.ml.function.SingleNodeObjectiveFunctionTest]].
 */
class LogisticLossFunctionTest {

  @Test
  def testCalculate(): Unit = {
    val delta = MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD
    val features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
    val coefficients = DenseVector[Double](1.0, 12.3, -21.0, 0.0)
    val offset = 1.5
    val margin = offset + features.dot(coefficients)
    val positiveLabel = 1D
    val negativeLabel = 0D

    // Test positive label
    val (value1, _) = LogisticLossFunction.lossAndDzLoss(margin, positiveLabel)
    // Compute the expected value by explicit computation
    val expected1 = math.log(1 + math.exp(-margin))
    assertEquals(value1, expected1, delta)

    // Test negative label
    val (value2, _) = LogisticLossFunction.lossAndDzLoss(margin, negativeLabel)
    val expected2 = math.log(1 + math.exp(margin))
    assertEquals(value2, expected2, delta)
  }

  @Test
  def testGradient(): Unit = {
    val delta = MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD
    val features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
    val coefficients = DenseVector[Double](1.0, 1.0, 1.0, 1.0)
    val offset = 0D
    val margin = offset + features.dot(coefficients)
    val positiveLabel = 1D
    val negativeLabel = 0D

    // Test positive label
    val (_, gradient1) = LogisticLossFunction.lossAndDzLoss(margin, positiveLabel)
    // Calculate gradient explicitly
    val expected1 = -1.0 / (1.0 + math.exp(margin))
    assertEquals(gradient1, expected1, delta)

    // Test negative label
    val expected2 = 1.0 / (1.0 + math.exp(-margin))
    val (_, gradient2) = LogisticLossFunction.lossAndDzLoss(margin, negativeLabel)
    assertEquals(gradient2, expected2, delta)
  }

  @Test
  def testHessianVector(): Unit = {
    val delta = MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD
    val label = 1D
    val offset = 0D

    // Test 0 vectors
    val features1 = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    val coefficients1 = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    val margin1 = offset + features1.dot(coefficients1)
    val sigma1 = 1.0 / (1.0 + math.exp(-margin1))
    val expected1 = sigma1 * (1 - sigma1)
    val D_1 = LogisticLossFunction.DzzLoss(margin1, label)
    assertEquals(D_1, expected1, delta)

    // Test non-zero vectors
    val features2 = DenseVector[Double](1.0, 0.0, 0.0, 0.0)
    val coefficients2 = DenseVector[Double](1.0, 0.0, 0.0, 0.0)
    val margin2 = offset + features2.dot(coefficients2)
    val sigma2 = 1.0 / (1.0 + math.exp(-margin2))
    val expected2 = sigma2 * (1 - sigma2)
    val D_2 = LogisticLossFunction.DzzLoss(margin2, label)
    assertEquals(D_2, expected2, delta)
  }
}
