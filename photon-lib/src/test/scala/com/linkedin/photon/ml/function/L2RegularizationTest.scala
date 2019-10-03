/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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

import breeze.linalg.{DenseMatrix, DenseVector, Vector, diag}
import org.mockito.Mockito._
import org.testng.annotations.Test
import org.testng.Assert.assertEquals

import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * Unit tests for [[L2Regularization]], [[L2RegularizationDiff]], and [[L2RegularizationTwiceDiff]].
 */
class L2RegularizationTest {

  import L2RegularizationTest._

  /**
   * Test that the L2 regularization mixin traits can correctly modify the existing behaviour of an objective function.
   */
  @Test
  def testAll(): Unit = {

    val mockNormalization = mock(classOf[BroadcastWrapper[NormalizationContext]])

    val regularizationWeight = 10D
    val coefficients = DenseVector.ones[Double](DIMENSION)
    val multiplyVector = coefficients * 2D
    val mockInterceptIndexOpt = Some(INTERCEPT_INDEX)

    val mockObjectiveFunction = new MockObjectiveFunction with L2RegularizationTwiceDiff {
      l2RegWeight = regularizationWeight
    }

    val mockObjectiveFunctionWithInterceptIndex = new MockObjectiveFunction with L2Regularization {
      l2RegWeight = regularizationWeight

      override def interceptOpt: Option[Int] = mockInterceptIndexOpt
    }

    // Assume that coefficients = 1-vector, multiply = 2-vector for all expected values below
    val expectedValue = MockObjectiveFunction.VALUE + (regularizationWeight * DIMENSION / 2)
    val expectedValueWithoutIntercept = MockObjectiveFunction.VALUE + (regularizationWeight * (DIMENSION - 1) / 2)
    val expectedGradient = DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.GRADIENT + regularizationWeight))
    val expectedVector =
      DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_VECTOR + (2D * regularizationWeight)))
    val expectedDiagonal =
      DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_DIAGONAL + regularizationWeight))
    val expectedMatrix =
      diag(DenseVector(Array.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_MATRIX + regularizationWeight)))

    assertEquals(mockObjectiveFunction.value(Unit, coefficients, mockNormalization), expectedValue)
    assertEquals(
      mockObjectiveFunctionWithInterceptIndex.value(Unit, coefficients, mockNormalization),
      expectedValueWithoutIntercept)
    assertEquals(mockObjectiveFunction.gradient(Unit, coefficients, mockNormalization), expectedGradient)
    assertEquals(
      mockObjectiveFunction.hessianVector(Unit, coefficients, multiplyVector, mockNormalization),
      expectedVector)
    assertEquals(mockObjectiveFunction.hessianDiagonal(Unit, coefficients), expectedDiagonal)
    assertEquals(mockObjectiveFunction.hessianMatrix(Unit, coefficients), expectedMatrix)
  }
}

object L2RegularizationTest {

  private val DIMENSION = 4
  private val INTERCEPT_INDEX = 1

  /**
   * Mock [[ObjectiveFunction]] for testing [[L2Regularization]].
   */
  class MockObjectiveFunction extends ObjectiveFunction with TwiceDiffFunction {

    import MockObjectiveFunction._

    type Data = Unit
    type Coefficients = Vector[Double]

    override protected[ml] def domainDimension(input: Unit): Int = 0

    override protected[ml] def value(
        input: Data,
        coefficients: Coefficients,
        normalizationContext: BroadcastWrapper[NormalizationContext]): Double =
      calculate(input, coefficients, normalizationContext)._1

    override protected[ml] def gradient(
        input: Data,
        coefficients: Coefficients,
        normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
      calculate(input, coefficients, normalizationContext)._2

    override protected[ml] def calculate(
        input: Data,
        coefficients: Coefficients,
        normalizationContext: BroadcastWrapper[NormalizationContext]): (Double, Vector[Double]) =
      (VALUE, DenseVector(Array.fill(coefficients.length)(GRADIENT)))

    override protected[ml] def hessianVector(
        input: Data,
        coefficients: Coefficients,
        multiplyVector: Coefficients,
        normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
      DenseVector(Array.fill(coefficients.length)(HESSIAN_VECTOR))

    override protected[ml] def hessianDiagonal(input: Data, coefficients: Coefficients): Vector[Double] =
      DenseVector(Array.fill(coefficients.length)(HESSIAN_DIAGONAL))

    override protected[ml] def hessianMatrix(input: Data, coefficients: Coefficients): DenseMatrix[Double] =
      diag(DenseVector(Array.fill(coefficients.length)(HESSIAN_MATRIX)))

    override protected[ml] def convertToVector(coefficients: Coefficients): Vector[Double] = coefficients

    override protected[ml] def convertFromVector(coefficients: Vector[Double]): Coefficients = coefficients
  }

  object MockObjectiveFunction {

    val VALUE = 0D
    val GRADIENT = 1D
    val HESSIAN_VECTOR = 2D
    val HESSIAN_DIAGONAL = 3D
    val HESSIAN_MATRIX = 4D
  }
}
