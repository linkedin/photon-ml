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
import com.linkedin.photon.ml.function.glm.LogisticLossFunction
import com.linkedin.photon.ml.function.{DistributedObjectiveFunction, L2RegularizationTwiceDiff}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.optimization.game.FixedEffectOptimizationConfiguration
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.CommonTestUtils
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * Unit tests for [[DistributedOptimizationProblem]].
 */
class DistributedOptimizationProblemTest {

  import DistributedOptimizationProblemTest._

  private val DIMENSIONS: Int = 5

  /**
   * Input data for testing correct routing of variance computation.
   */
  @DataProvider
  def varianceInput(): Array[Array[Any]] = {

    val mockTwiceDiffFunction = mock(classOf[DistributedObjectiveFunction])
    val mockOptimizerTwiceDiff = mock(classOf[Optimizer[DistributedObjectiveFunction]])
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
    val objectiveFunction = mock(classOf[DistributedObjectiveFunction])
    val optimizer = mock(classOf[Optimizer[DistributedObjectiveFunction]])
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
    doReturn((means, statesTracker, None)).when(optimizer).optimize(objectiveFunction, means)(trainingData)
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

    val (model, _) = problem.run(trainingData, initialModel)

    assertTrue(means.eq(model.coefficients.means))
  }

  /**
   * Test that regularization weights can be updated.
   */
  @Test
  def testUpdateRegularizationWeight(): Unit = {

    val normalization = NoNormalization()
    val initL1Weight = 1D
    val initL2Weight = 2D
    val finalL1Weight = 3D
    val finalL2Weight = 4D
    val finalElasticWeight = 5D
    val alpha = 0.75
    val elasticFinalL1Weight = finalElasticWeight * alpha
    val elasticFinalL2Weight = finalElasticWeight * (1 - alpha)

    val normalizationMock = mock(classOf[BroadcastWrapper[NormalizationContext]])
    val optimizer = mock(classOf[Optimizer[DistributedObjectiveFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val objectiveFunction = mock(classOf[DistributedObjectiveFunction])

    doReturn(normalization).when(normalizationMock).value
    doReturn(statesTracker).when(optimizer).getStateTracker

    val optimizerL1 = new OWLQN(initL1Weight, normalizationMock)
    val objectiveFunctionL2 = MOCK_DISTRIBUTED_OBJECTIVE_FUNCTION
    objectiveFunctionL2.l2RegularizationWeight = initL2Weight

    val l1Problem = new DistributedOptimizationProblem(
      optimizerL1,
      objectiveFunction,
      samplerOption = None,
      LogisticRegressionModel.apply,
      L1RegularizationContext,
      VarianceComputationType.NONE)
    val l2Problem = new DistributedOptimizationProblem(
      optimizer,
      objectiveFunctionL2,
      samplerOption = None,
      LogisticRegressionModel.apply,
      L2RegularizationContext,
      VarianceComputationType.NONE)
    val elasticProblem = new DistributedOptimizationProblem(
      optimizerL1,
      objectiveFunctionL2,
      samplerOption = None,
      LogisticRegressionModel.apply,
      ElasticNetRegularizationContext(alpha),
      VarianceComputationType.NONE)

    // Check update to L1/L2 weights individually
    assertNotEquals(optimizerL1.l1RegularizationWeight, finalL1Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertNotEquals(objectiveFunctionL2.l2RegularizationWeight, finalL2Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(optimizerL1.l1RegularizationWeight, initL1Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(objectiveFunctionL2.l2RegularizationWeight, initL2Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)

    l1Problem.updateRegularizationWeight(finalL1Weight)
    l2Problem.updateRegularizationWeight(finalL2Weight)

    assertNotEquals(optimizerL1.l1RegularizationWeight, initL1Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertNotEquals(objectiveFunctionL2.l2RegularizationWeight, initL2Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(optimizerL1.l1RegularizationWeight, finalL1Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(objectiveFunctionL2.l2RegularizationWeight, finalL2Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)

    // Check updates to L1/L2 weights together
    optimizerL1.l1RegularizationWeight = initL1Weight
    objectiveFunctionL2.l2RegularizationWeight = initL2Weight

    assertNotEquals(optimizerL1.l1RegularizationWeight, elasticFinalL1Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertNotEquals(objectiveFunctionL2.l2RegularizationWeight, elasticFinalL2Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(optimizerL1.l1RegularizationWeight, initL1Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(objectiveFunctionL2.l2RegularizationWeight, initL2Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)

    elasticProblem.updateRegularizationWeight(finalElasticWeight)

    assertNotEquals(optimizerL1.l1RegularizationWeight, initL1Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertNotEquals(objectiveFunctionL2.l2RegularizationWeight, initL2Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(optimizerL1.l1RegularizationWeight, elasticFinalL1Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(objectiveFunctionL2.l2RegularizationWeight, elasticFinalL2Weight, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }
}

object DistributedOptimizationProblemTest {

  type L2DistributedObjectiveFunction = DistributedObjectiveFunction with L2RegularizationTwiceDiff

  private val MOCK_OPTIMIZER_CONFIG = mock(classOf[OptimizerConfig])
  private val MOCK_COORDINATE_CONFIG = FixedEffectOptimizationConfiguration(
    MOCK_OPTIMIZER_CONFIG,
    L2RegularizationContext)
  private val MOCK_DISTRIBUTED_OBJECTIVE_FUNCTION = DistributedObjectiveFunction(
      MOCK_COORDINATE_CONFIG,
      LogisticLossFunction,
      treeAggregateDepth = 1,
      interceptIndexOpt = None)
    .asInstanceOf[DistributedObjectiveFunction with L2RegularizationTwiceDiff]
}
