/*
 * Copyright 2020 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.function

import breeze.linalg.{DenseVector, diag}
import org.testng.annotations.Test
import org.testng.Assert.assertEquals

import com.linkedin.photon.ml.model.{Coefficients => ModelCoefficients}
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.util.PhotonNonBroadcast

/**
 * Unit tests for [[PriorDistribution]], [[PriorDistributionDiff]], and [[PriorDistributionTwiceDiff]].
 */
class PriorDistributionTest {

  import L2RegularizationTest._

  private val DIMENSION = 4

  /**
   * Test that the prior distribution mixin traits can correctly modify the existing behaviour of an objective function.
   */
  @Test
  def testAll(): Unit = {

    val noNormalizationBroadcast = PhotonNonBroadcast(NoNormalization())

    val coefficients = DenseVector.ones[Double](DIMENSION)
    val priorMean = coefficients *:* 2D
    val multiplyVector = coefficients *:* 3D
    val priorVar = coefficients *:* 4D
    val normalizationContext = PhotonNonBroadcast(new NormalizationContext(Some(coefficients *:* 2D), None))

    val increWeight = 10D

    val mockObjectiveFunction = new MockObjectiveFunction with PriorDistributionTwiceDiff {
      override val priorCoefficients = ModelCoefficients(priorMean, Option(priorVar))
      incrementalWeight = increWeight
    }

    /**
     * Assume that coefficients = 1-vector, prior mean = 2-vector, multiply = 3-vector, prior variance = 4-vector for all expected values below
     *
     * l2RegValue = sum(DenseVector.fill(DIMENSION){pow(1 - 2, 2) / 4)}) * increWeight / 2 = 0.25 * increWeight * DIMENSION / 2;
     * l2RegGradient = (1 - 2) / 4 * increWeight = (-0.25) * increWeight;
     * l2RegHessianDiagonal = 1 / 4 * increWeight = 0.25 * increWeight;
     * l2RegHessianVector = 3 / 4 * increWeight = 0.75 * increWeight;
     * l2RegNormalizedValue = sum(DenseVector.fill(DIMENSION){pow(1 - 2 / 2, 2) / 1)}) * increWeight / 2 =  0;
     * l2RegNormalizedGradient = (1 - 2 / 2) / 1 * increWeight = 0;
     * l2RegHessianVector = 3 / 1 * increWeight = 3 * increWeight.
     */
    val expectedValue = MockObjectiveFunction.VALUE + 0.25 * increWeight * DIMENSION / 2
    val expectedGradient = DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.GRADIENT + (-0.25) * increWeight))
    val expectedVector = DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_VECTOR + 0.75 * increWeight))
    val expectedDiagonal = DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_DIAGONAL + 0.25 * increWeight))
    val expectedMatrix = diag(DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_MATRIX + 0.25 * increWeight)))
    val expectedNormalizedValue = MockObjectiveFunction.VALUE
    val expectedNormalizedGradient = DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.GRADIENT))
    val expectedNormalizedVector = DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_VECTOR + 3 * increWeight))

    assertEquals(mockObjectiveFunction.value(Unit, coefficients, noNormalizationBroadcast), expectedValue)
    assertEquals(mockObjectiveFunction.gradient(Unit, coefficients, noNormalizationBroadcast), expectedGradient)
    assertEquals(
      mockObjectiveFunction.hessianVector(Unit, coefficients, multiplyVector, noNormalizationBroadcast),
      expectedVector)
    assertEquals(mockObjectiveFunction.hessianDiagonal(Unit, coefficients), expectedDiagonal)
    assertEquals(mockObjectiveFunction.hessianMatrix(Unit, coefficients), expectedMatrix)
    assertEquals(mockObjectiveFunction.value(Unit, coefficients, normalizationContext), expectedNormalizedValue)
    assertEquals(mockObjectiveFunction.gradient(Unit, coefficients, normalizationContext), expectedNormalizedGradient)
    assertEquals(
      mockObjectiveFunction.hessianVector(Unit, coefficients, multiplyVector, normalizationContext),
      expectedNormalizedVector)
  }
}
