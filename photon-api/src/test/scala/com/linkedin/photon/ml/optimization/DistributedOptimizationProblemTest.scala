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
package com.linkedin.photon.ml.optimization

import breeze.linalg.{DenseMatrix, DenseVector, Vector}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.DistributedObjectiveFunction
import com.linkedin.photon.ml.function.glm.DistributedGLMLossFunction
import com.linkedin.photon.ml.function.svm.DistributedSmoothedHingeLossFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * Unit tests for [[DistributedOptimizationProblem]].
 */
class DistributedOptimizationProblemTest {

  private val DIMENSIONS: Int = 5

  /**
   * Input data for testing correct routing of variance computation.
   */
  @DataProvider
  def varianceInput(): Array[Array[Any]] = {

    val mockDiffFunction = mock(classOf[DistributedSmoothedHingeLossFunction])
    val mockTwiceDiffFunction = mock(classOf[DistributedGLMLossFunction])
    val mockOptimizerDiff = mock(classOf[Optimizer[DistributedSmoothedHingeLossFunction]])
    val mockOptimizerTwiceDiff = mock(classOf[Optimizer[DistributedGLMLossFunction]])
    val mockStatesTracker = mock(classOf[OptimizationStatesTracker])

    val hessianDiagonal = DenseVector(Array(1D, 0D, 2D))
    val hessianMatrix = DenseMatrix.eye[Double](DIMENSIONS)

    doReturn(mockStatesTracker).when(mockOptimizerDiff).getStateTracker
    doReturn(mockStatesTracker).when(mockOptimizerTwiceDiff).getStateTracker
    doReturn(hessianDiagonal)
      .when(mockTwiceDiffFunction)
      .hessianDiagonal(Matchers.any(), Matchers.any())
    doReturn(hessianMatrix)
      .when(mockTwiceDiffFunction)
      .hessianMatrix(Matchers.any(), Matchers.any())

    val matrixVariance = DenseMatrix.eye[Double](DIMENSIONS)

    Array(
      // var type, function, expected result
      Array(VarianceComputationType.NONE, mockOptimizerDiff, mockDiffFunction, None),
      Array(VarianceComputationType.NONE, mockOptimizerTwiceDiff, mockTwiceDiffFunction, None),
      Array(VarianceComputationType.SIMPLE, mockOptimizerDiff, mockDiffFunction, None),
      Array(VarianceComputationType.SIMPLE, mockOptimizerTwiceDiff, mockTwiceDiffFunction, None),
      Array(VarianceComputationType.FULL, mockOptimizerDiff, mockDiffFunction, None),
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
      optimizer: Optimizer[DistributedObjectiveFunction],
      objectiveFunction: DistributedObjectiveFunction,
      expected: Option[Vector[Double]]): Unit = {

    val mockSparkContext = mock(classOf[SparkContext])
    val mockTrainingData = mock(classOf[RDD[LabeledPoint]])
    val mockCoefficients = mock(classOf[Vector[Double]])
    val mockBroadcast = mock(classOf[Broadcast[Vector[Double]]])

    doReturn(mockSparkContext).when(mockTrainingData).sparkContext
    doReturn(mockBroadcast).when(mockSparkContext).broadcast(mockCoefficients)
    doReturn(mockCoefficients).when(mockBroadcast).value

    val problem = new DistributedOptimizationProblem(
      optimizer,
      objectiveFunction,
      samplerOption = None,
      LogisticRegressionModel.apply,
      NoRegularizationContext,
      varianceComputationType)

    assertEquals(problem.computeVariances(mockTrainingData, mockCoefficients), expected)
  }

  /**
   * Test a mock run-through of the optimization problem.
   */
  @Test
  def testRun(): Unit = {

    val normalization = NoNormalization()

    val sparkContext = mock(classOf[SparkContext])
    val trainingData = mock(classOf[RDD[LabeledPoint]])
    val objectiveFunction = mock(classOf[DistributedGLMLossFunction])
    val optimizer = mock(classOf[Optimizer[DistributedGLMLossFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val state = mock(classOf[OptimizerState])
    val broadcastCoefficients = mock(classOf[Broadcast[Vector[Double]]])
    val broadcastNormalization = mock(classOf[BroadcastWrapper[NormalizationContext]])
    val initialModel = mock(classOf[GeneralizedLinearModel])
    val coefficients = mock(classOf[Coefficients])
    val means = mock(classOf[Vector[Double]])

    doReturn(sparkContext).when(trainingData).sparkContext
    doReturn(broadcastCoefficients).when(sparkContext).broadcast(means)
    doReturn(broadcastNormalization).when(optimizer).getNormalizationContext
    doReturn(normalization).when(broadcastNormalization).value
    doReturn((means, None)).when(optimizer).optimize(objectiveFunction, means)(trainingData)
    doReturn(statesTracker).when(optimizer).getStateTracker
    doReturn(Array(state)).when(statesTracker).getTrackedStates
    doReturn(means).when(state).coefficients
    doReturn(coefficients).when(initialModel).coefficients
    doReturn(means).when(coefficients).means

    val problem = new DistributedOptimizationProblem(
      optimizer,
      objectiveFunction,
      samplerOption = None,
      LogisticRegressionModel.apply,
      NoRegularizationContext,
      VarianceComputationType.NONE)

    val model = problem.run(trainingData, initialModel)

    assertTrue(means.eq(model.coefficients.means))
  }
}
