///*
// * Copyright 2019 LinkedIn Corp. All rights reserved.
// * Licensed under the Apache License, Version 2.0 (the "License"); you may
// * not use this file except in compliance with the License. You may obtain a
// * copy of the License at
// *
// * http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations
// * under the License.
// */
//package com.linkedin.photon.ml.function
//
//import breeze.linalg.{DenseVector, diag}
//import org.testng.annotations.Test
//import org.testng.Assert.assertEquals
//import org.mockito.Mockito.mock
//
//import com.linkedin.photon.ml.model.{Coefficients => ModelCoefficients}
//import com.linkedin.photon.ml.normalization.NormalizationContext
//import com.linkedin.photon.ml.util.BroadcastWrapper
//
///**
// * Unit tests for [[PriorDistribution]], [[PriorDistributionDiff]], and [[PriorDistributionTwiceDiff]].
// */
//class PriorDistributionTest {
//
//  import L2RegularizationTest._
//
//  private val DIMENSION = 4
//
//  /**
//   * Test that the prior distribution mixin traits can correctly modify the existing behaviour of an objective function.
//   */
//  @Test
//  def testAll(): Unit = {
//
//    val mockNormalization = mock(classOf[BroadcastWrapper[NormalizationContext]])
//
//    val coefficients = DenseVector.ones[Double](DIMENSION)
//    val priorMean = coefficients :* 2D
//    val multiplyVector = coefficients * 3D
//    val priorVar = coefficients :* 4D
//
//    val l2Weight = 10D
//
//    val mockObjectiveFunction = new MockObjectiveFunction with PriorDistributionTwiceDiff {
//      override val priorCoefficients = ModelCoefficients(priorMean, Option(priorVar))
//      l2RegWeight = l2Weight
//    }
//
//    /**
//     * Assume that coefficients = 1-vector, prior mean = 2-vector, multiply = 3-vector, prior variance = 4-vector for all expected values below
//     *
//     * l2RegValue = sum(DenseVector.fill(DIMENSION){pow(1 - 2, 2) / 4)}) * l2Weight / 2 = 0.25 * l2Weight * DIMENSION / 2;
//     * l2RegGradient = (1 - 2) / 4 * l2Weight = (-0.25) * l2Weight;
//     * l2RegHessianDiagonal = 1 / 4 * l2Weight = 0.25 * l2Weight;
//     * l2RegHessianVector = 3 / 4 * l2Weight = 0.75 * l2Weight.
//     */
//    val expectedValue = MockObjectiveFunction.VALUE + 0.25 * l2Weight * DIMENSION / 2
//    val expectedGradient = DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.GRADIENT + (-0.25) * l2Weight))
//    val expectedVector = DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_VECTOR + 0.75 * l2Weight))
//    val expectedDiagonal = DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_DIAGONAL + 0.25 * l2Weight))
//    val expectedMatrix = diag(DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_MATRIX + 0.25 * l2Weight)))
//
//    assertEquals(mockObjectiveFunction.value(Unit, coefficients, mockNormalization), expectedValue)
//    assertEquals(mockObjectiveFunction.gradient(Unit, coefficients, mockNormalization), expectedGradient)
//    assertEquals(
//      mockObjectiveFunction.hessianVector(Unit, coefficients, multiplyVector, mockNormalization),
//      expectedVector)
//    assertEquals(mockObjectiveFunction.hessianDiagonal(Unit, coefficients), expectedDiagonal)
//    assertEquals(mockObjectiveFunction.hessianMatrix(Unit, coefficients), expectedMatrix)
//  }
//}
