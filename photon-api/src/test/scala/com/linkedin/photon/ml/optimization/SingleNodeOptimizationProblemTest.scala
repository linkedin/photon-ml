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
package com.linkedin.photon.ml.optimization

import breeze.linalg.{DenseMatrix, DenseVector, Vector}
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * Unit tests for [[SingleNodeOptimizationProblem]].
 */
class SingleNodeOptimizationProblemTest {

  private val DIMENSIONS: Int = 5

  /**
   * Input data for testing correct routing of variance computation.
   */
  @DataProvider
  def varianceInput(): Array[Array[Any]] = {

    val mockTwiceDiffFunction = mock(classOf[SingleNodeObjectiveFunction])
    val mockOptimizerTwiceDiff = mock(classOf[Optimizer[SingleNodeObjectiveFunction]])
    val mockStatesTracker = mock(classOf[OptimizationStatesTracker])

    val hessianDiagonal = DenseVector(Array(1D, 0D, 2D))
    val hessianMatrix = DenseMatrix.eye[Double](DIMENSIONS)

    doReturn(mockStatesTracker).when(mockOptimizerTwiceDiff).getStateTracker
    doReturn(hessianDiagonal)
      .when(mockTwiceDiffFunction)
      .hessianDiagonal(Matchers.any(), Matchers.any())
    doReturn(hessianMatrix)
      .when(mockTwiceDiffFunction)
      .hessianMatrix(Matchers.any(), Matchers.any())

    val diagonalVariance = DenseVector(Array(1D, 1D / MathConst.EPSILON, 0.5))
    val matrixVariance = DenseVector(Array.fill(DIMENSIONS)(1D))

    Array(
      Array(VarianceComputationType.NONE, mockOptimizerTwiceDiff, mockTwiceDiffFunction, None),
      Array(VarianceComputationType.SIMPLE, mockOptimizerTwiceDiff, mockTwiceDiffFunction, Some(diagonalVariance)),
      Array(VarianceComputationType.FULL, mockOptimizerTwiceDiff, mockTwiceDiffFunction, Some(matrixVariance)))
  }

  /**
   * Test that the variance computation is correctly routed based on the objective function and type of variance
   * computation selected.
   *
   * @param varianceComputationType If an how to compute coefficient variances
   * @param optimizer The underlying optimizer which iteratively solves the convex problem
   * @param objectiveFunction The objective function to optimize
   * @param expected The expected result
   */
  @Test(dataProvider = "varianceInput")
  def testComputeVariances(
      varianceComputationType: VarianceComputationType,
      optimizer: Optimizer[SingleNodeObjectiveFunction],
      objectiveFunction: SingleNodeObjectiveFunction,
      expected: Option[Vector[Double]]): Unit = {

    val problem = new SingleNodeOptimizationProblem(
      optimizer,
      objectiveFunction,
      LogisticRegressionModel.apply,
      varianceComputationType)
    val trainingData = mock(classOf[Iterable[LabeledPoint]])
    val coefficients = mock(classOf[Vector[Double]])

    assertEquals(problem.computeVariances(trainingData, coefficients), expected)
  }

  /**
   * Test a mock run-through of the optimization problem.
   */
  @Test
  def testRun(): Unit = {

    val normalization = NoNormalization()

    val trainingData = mock(classOf[Iterable[LabeledPoint]])
    val objectiveFunction = mock(classOf[SingleNodeObjectiveFunction])
    val optimizer = mock(classOf[Optimizer[SingleNodeObjectiveFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val state = mock(classOf[OptimizerState])
    val broadcastNormalization = mock(classOf[BroadcastWrapper[NormalizationContext]])
    val initialModel = mock(classOf[GeneralizedLinearModel])
    val coefficients = mock(classOf[Coefficients])
    val means = mock(classOf[Vector[Double]])

    doReturn(broadcastNormalization).when(optimizer).getNormalizationContext
    doReturn(normalization).when(broadcastNormalization).value
    doReturn((means, statesTracker, None)).when(optimizer).optimize(objectiveFunction, means)(trainingData)
    doReturn(Array(state)).when(statesTracker).getTrackedStates
    doReturn(means).when(state).coefficients
    doReturn(coefficients).when(initialModel).coefficients
    doReturn(means).when(coefficients).means

    val problem = new SingleNodeOptimizationProblem(
      optimizer,
      objectiveFunction,
      LogisticRegressionModel.apply,
      VarianceComputationType.NONE)

    val (model, _) = problem.run(trainingData, initialModel)

    assertTrue(means.eq(model.coefficients.means))
  }
}
