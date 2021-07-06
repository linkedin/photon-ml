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

import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, Vector, diag}
import org.mockito.Mockito._
import org.testng.annotations.Test
import org.testng.Assert.assertEquals

import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.TestObjective
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

    val mockObjectiveFunctionWithIntercept = new MockObjectiveFunction with L2RegularizationTwiceDiff {
      l2RegWeight = regularizationWeight

      override def interceptOpt: Option[Int] = mockInterceptIndexOpt
    }

    // L2 norm of 1-vector of dimension d is d
    val expectedValue = MockObjectiveFunction.VALUE + (regularizationWeight * DIMENSION / 2)
    val expectedValueWithIntercept = MockObjectiveFunction.VALUE + (regularizationWeight * (DIMENSION - 1) / 2)

    // L2 gradient if coefficients * L2 weight, but in this case all coefficients are 1
    val expectedGradient = DenseVector.fill(DIMENSION)(MockObjectiveFunction.GRADIENT + regularizationWeight)
    val expectedGradientWithIntercept =
      expectedGradient -:- SparseVector[Double](DIMENSION)((INTERCEPT_INDEX, regularizationWeight))

    // L2 gradient of (H * v) is (v * L2 weight), and in this case v_i is 2 for all i
    val expectedVector =
      DenseVector.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_VECTOR + (2D * regularizationWeight))
    val expectedVectorWithIntercept =
      expectedVector -:- SparseVector[Double](DIMENSION)((INTERCEPT_INDEX, 2D * regularizationWeight))

    val expectedDiagonal =
      DenseVector.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_DIAGONAL + regularizationWeight)
    val expectedDiagonalWithIntercept =
      expectedDiagonal -:- SparseVector[Double](DIMENSION)((INTERCEPT_INDEX, regularizationWeight))

    val expectedMatrix = diag(DenseVector.fill(DIMENSION)(MockObjectiveFunction.HESSIAN_MATRIX + regularizationWeight))
    val expectedMatrixWithIntercept =
      expectedMatrix -:- diag(SparseVector[Double](DIMENSION)((INTERCEPT_INDEX, regularizationWeight)))

    assertEquals(mockObjectiveFunction.value(Unit, coefficients, mockNormalization), expectedValue)
    assertEquals(
      mockObjectiveFunctionWithIntercept.value(Unit, coefficients, mockNormalization),
      expectedValueWithIntercept)

    assertEquals(mockObjectiveFunction.gradient(Unit, coefficients, mockNormalization), expectedGradient)
    assertEquals(
      mockObjectiveFunctionWithIntercept.gradient(Unit, coefficients, mockNormalization),
      expectedGradientWithIntercept)

    assertEquals(
      mockObjectiveFunction.hessianVector(Unit, coefficients, multiplyVector, mockNormalization),
      expectedVector)
    assertEquals(
      mockObjectiveFunctionWithIntercept.hessianVector(Unit, coefficients, multiplyVector, mockNormalization),
      expectedVectorWithIntercept)

    assertEquals(mockObjectiveFunction.hessianDiagonal(Unit, coefficients), expectedDiagonal)
    assertEquals(mockObjectiveFunctionWithIntercept.hessianDiagonal(Unit, coefficients), expectedDiagonalWithIntercept)

    assertEquals(mockObjectiveFunction.hessianMatrix(Unit, coefficients), expectedMatrix)
    assertEquals(mockObjectiveFunctionWithIntercept.hessianMatrix(Unit, coefficients), expectedMatrixWithIntercept)
  }
}

object L2RegularizationTest {

  private val DIMENSION = 4
  private val INTERCEPT_INDEX = 1

  /**
   * Mock [[ObjectiveFunction]] for testing [[L2Regularization]].
   */
  class MockObjectiveFunction
    extends ObjectiveFunction(new TestObjective.MockPointwiseLossFunction)
      with TwiceDiffFunction {

    import MockObjectiveFunction._

    type Data = Unit

    override protected[ml] def value(
        input: Data,
        coefficients: Vector[Double],
        normalizationContext: BroadcastWrapper[NormalizationContext]): Double =
      VALUE

    override protected[ml] def gradient(
        input: Data,
        coefficients: Vector[Double],
        normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
      DenseVector.fill(coefficients.length)(GRADIENT)

    override protected[ml] def calculate(
        input: Data,
        coefficients: Vector[Double],
        normalizationContext: BroadcastWrapper[NormalizationContext]): (Double, Vector[Double]) =
      (VALUE, DenseVector.fill(coefficients.length)(GRADIENT))

    override protected[ml] def hessianVector(
        input: Data,
        coefficients: Vector[Double],
        multiplyVector: Vector[Double],
        normalizationContext: BroadcastWrapper[NormalizationContext]): Vector[Double] =
      DenseVector.fill(coefficients.length)(HESSIAN_VECTOR)

    override protected[ml] def hessianDiagonal(input: Data, coefficients: Vector[Double]): Vector[Double] =
      DenseVector.fill(coefficients.length)(HESSIAN_DIAGONAL)

    override protected[ml] def hessianMatrix(input: Data, coefficients: Vector[Double]): DenseMatrix[Double] =
      diag(DenseVector.fill(coefficients.length)(HESSIAN_MATRIX))
  }

  object MockObjectiveFunction {

    val VALUE = 0D
    val GRADIENT = 1D
    val HESSIAN_VECTOR = 2D
    val HESSIAN_DIAGONAL = 3D
    val HESSIAN_MATRIX = 4D
  }
}
