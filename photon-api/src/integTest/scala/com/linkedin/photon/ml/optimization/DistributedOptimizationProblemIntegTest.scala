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

import java.util.Random

import breeze.linalg.{DenseMatrix, DenseVector, Vector, diag, pinv}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.L2RegularizationDiff
import com.linkedin.photon.ml.function.glm._
import com.linkedin.photon.ml.function.svm.DistributedSmoothedHingeLossFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.optimization.game.FixedEffectOptimizationConfiguration
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}
import com.linkedin.photon.ml.util.{BroadcastWrapper, VectorUtils}

/**
 * Integration tests for [[DistributedOptimizationProblem]].
 */
class DistributedOptimizationProblemIntegTest extends SparkTestUtils {

  import CommonTestUtils._
  import DistributedOptimizationProblemIntegTest._

  /**
   * Function to generate a mock [[GeneralizedLinearModel]].
   *
   * @param coefficients Model coefficients (unused)
   * @return A mocked [[GeneralizedLinearModel]]
   */
  def glmConstructorMock(coefficients: Coefficients): GeneralizedLinearModel = mock(classOf[GeneralizedLinearModel])

  /**
   * Generate weighted benign datasets for binary classification.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateWeightedBenignDatasetBinaryClassification: Seq[LabeledPoint] = {

    val r = new Random(OptimizationProblemIntegTestUtils.WEIGHT_RANDOM_SEED)

    drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      OptimizationProblemIntegTestUtils.DATA_RANDOM_SEED,
      OptimizationProblemIntegTestUtils.TRAINING_SAMPLES,
      OptimizationProblemIntegTestUtils.DIMENSIONS)
      .map { obj =>
        assertEquals(obj._2.length, OptimizationProblemIntegTestUtils.DIMENSIONS, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * OptimizationProblemIntegTestUtils.WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      }
      .toList
  }

  /**
   * Generate weighted benign datasets for linear regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateWeightedBenignDatasetLinearRegression: Seq[LabeledPoint] = {

    val r = new Random(OptimizationProblemIntegTestUtils.WEIGHT_RANDOM_SEED)

    drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(
      OptimizationProblemIntegTestUtils.DATA_RANDOM_SEED,
      OptimizationProblemIntegTestUtils.TRAINING_SAMPLES,
      OptimizationProblemIntegTestUtils.DIMENSIONS)
      .map { obj =>
        assertEquals(obj._2.length, OptimizationProblemIntegTestUtils.DIMENSIONS, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * OptimizationProblemIntegTestUtils.WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      }
      .toList
  }
  /**
   * Generate weighted benign datasets for Poisson regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateWeightedBenignDatasetPoissonRegression: Seq[LabeledPoint] = {

    val r = new Random(OptimizationProblemIntegTestUtils.WEIGHT_RANDOM_SEED)

    drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      OptimizationProblemIntegTestUtils.DATA_RANDOM_SEED,
      OptimizationProblemIntegTestUtils.TRAINING_SAMPLES,
      OptimizationProblemIntegTestUtils.DIMENSIONS)
      .map { obj =>
        assertEquals(obj._2.length, OptimizationProblemIntegTestUtils.DIMENSIONS, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * OptimizationProblemIntegTestUtils.WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      }
      .toList
  }

  @DataProvider(parallel = true)
  def varianceInput(): Array[Array[Any]] = {

    val regularizationWeights = Array[Double](0.1, 0.0, 1.0, 10.0, 100.0)

    // Regularization weight, input data generation function, objective function, manual Hessian calculation function
    regularizationWeights.flatMap { weight =>
      Array(
        Array[Any](
          weight,
          generateWeightedBenignDatasetBinaryClassification _,
          LogisticLossFunction,
          OptimizationProblemIntegTestUtils.logisticDzzLoss _),
        Array[Any](
          weight,
          generateWeightedBenignDatasetLinearRegression _,
          SquaredLossFunction,
          OptimizationProblemIntegTestUtils.linearDzzLoss _),
        Array[Any](
          weight,
          generateWeightedBenignDatasetPoissonRegression _,
          PoissonLossFunction,
          OptimizationProblemIntegTestUtils.poissonDzzLoss _))
    }
  }

  /**
   * Test that regularization weights can be updated.
   */
  @Test
  def testUpdateRegularizationWeight(): Unit = sparkTest("testUpdateRegularizationWeight") {

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
    val optimizer = mock(classOf[Optimizer[DistributedSmoothedHingeLossFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val objectiveFunction = mock(classOf[DistributedSmoothedHingeLossFunction])

    doReturn(normalization).when(normalizationMock).value
    doReturn(Some(statesTracker)).when(optimizer).getStateTracker

    val optimizerL1 = new OWLQN(initL1Weight, normalizationMock)
    val objectiveFunctionL2 = new L2LossFunction(sc)
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

  /**
   * Test simple coefficient variance computation for weighted data points, with regularization.
   *
   * @param regularizationWeight Regularization weight
   * @param dataGenerationFunction Function to generate test data
   * @param lossFunction Loss function for optimization
   * @param DzzLossFunction Function to compute coefficient Hessian directly
   */
  @Test(dataProvider = "varianceInput")
  def testComputeVariancesSimple(
      regularizationWeight: Double,
      dataGenerationFunction: () => Seq[LabeledPoint],
      lossFunction: PointwiseLossFunction,
      DzzLossFunction: (Vector[Double]) => (LabeledPoint) => Double): Unit = sparkTest("testComputeVariancesSimple") {

    val input = sc.parallelize(dataGenerationFunction())
    val coefficients = generateDenseVector(OptimizationProblemIntegTestUtils.DIMENSIONS)

    val optimizer = mock(classOf[Optimizer[DistributedGLMLossFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val regContext = mock(classOf[RegularizationContext])
    val optConfig = mock(classOf[FixedEffectOptimizationConfiguration])

    doReturn(Some(statesTracker)).when(optimizer).getStateTracker
    doReturn(regContext).when(optConfig).regularizationContext
    doReturn(regularizationWeight).when(optConfig).regularizationWeight
    doReturn(RegularizationType.L2).when(regContext).regularizationType
    doReturn(regularizationWeight).when(regContext).getL2RegularizationWeight(regularizationWeight)

    val objective = DistributedGLMLossFunction(optConfig, lossFunction, treeAggregateDepth = 1)

    val optimizationProblem = new DistributedOptimizationProblem(
      optimizer,
      objective,
      samplerOption = None,
      glmConstructorMock,
      NoRegularizationContext,
      VarianceComputationType.SIMPLE)

    val hessianDiagonal = input.treeAggregate(DenseVector.zeros[Double](OptimizationProblemIntegTestUtils.DIMENSIONS))(
      seqOp = (vector: DenseVector[Double], datum: LabeledPoint) => {
        diag(OptimizationProblemIntegTestUtils.hessianSum(DzzLossFunction(coefficients))(diag(vector), datum))
      },
      combOp = (vector1: DenseVector[Double], vector2: DenseVector[Double]) => vector1 + vector2,
      depth = 1)
    // Simple estimate of the diagonal of the covariance matrix (instead of a full inverse).
    val expected = (hessianDiagonal + regularizationWeight).map( v => 1D / (v + MathConst.EPSILON))
    val actual: Vector[Double] = optimizationProblem.computeVariances(input, coefficients).get

    assertTrue(VectorUtils.areAlmostEqual(actual, expected))
  }

  /**
   * Test full coefficient variance computation for weighted data points, with regularization.
   *
   * @param regularizationWeight Regularization weight
   * @param dataGenerationFunction Function to generate test data
   * @param lossFunction Loss function for optimization
   * @param DzzLossFunction Function to compute coefficient Hessian directly
   */
  @Test(dataProvider = "varianceInput")
  def testComputeVariancesFull(
      regularizationWeight: Double,
      dataGenerationFunction: () => Seq[LabeledPoint],
      lossFunction: PointwiseLossFunction,
      DzzLossFunction: (Vector[Double]) => (LabeledPoint) => Double): Unit = sparkTest("testComputeVariancesFull") {

    val input = sc.parallelize(dataGenerationFunction())
    val dimensions = OptimizationProblemIntegTestUtils.DIMENSIONS
    val coefficients = generateDenseVector(dimensions)

    val optimizer = mock(classOf[Optimizer[DistributedGLMLossFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val regContext = mock(classOf[RegularizationContext])
    val optConfig = mock(classOf[FixedEffectOptimizationConfiguration])

    doReturn(Some(statesTracker)).when(optimizer).getStateTracker
    doReturn(regContext).when(optConfig).regularizationContext
    doReturn(regularizationWeight).when(optConfig).regularizationWeight
    doReturn(RegularizationType.L2).when(regContext).regularizationType
    doReturn(regularizationWeight).when(regContext).getL2RegularizationWeight(regularizationWeight)

    val objective = DistributedGLMLossFunction(optConfig, lossFunction, treeAggregateDepth = 1)

    val optimizationProblem = new DistributedOptimizationProblem(
      optimizer,
      objective,
      samplerOption = None,
      glmConstructorMock,
      NoRegularizationContext,
      VarianceComputationType.FULL)

    val hessianMatrix = input.treeAggregate(
      DenseMatrix.zeros[Double](dimensions, dimensions))(
      seqOp = OptimizationProblemIntegTestUtils.hessianSum(DzzLossFunction(coefficients)),
      combOp = (matrix1: DenseMatrix[Double], matrix2: DenseMatrix[Double]) => matrix1 + matrix2,
      depth = 1)
    // Simple estimate of the diagonal of the covariance matrix (instead of a full inverse).
    val expected = diag(pinv(hessianMatrix + (DenseMatrix.eye[Double](dimensions) * regularizationWeight)))
    val actual: Vector[Double] = optimizationProblem.computeVariances(input, coefficients).get

    assertTrue(VectorUtils.areAlmostEqual(actual, expected))
  }

  /**
   * Test the variance computation against a reference implementation in R glm.
   */
  @Test
  def testComputeVariancesAgainstReference(): Unit = sparkTest("testComputeVariancesAgainstReference") {

    // Read the "heart disease" dataset from libSVM format
    val input: RDD[LabeledPoint] = {
      val tt = getClass.getClassLoader.getResource("DriverIntegTest/input/heart.txt")
      val inputFile = tt.toString
      val rawInput = sc.textFile(inputFile, 1)

      rawInput.map { x =>
        val y = x.split(" ")
        val label = y(0).toDouble / 2 + 0.5
        val features = y.drop(1).map(z => z.split(":")(1).toDouble) :+ 1.0
        new LabeledPoint(label, DenseVector(features))
      }
    }

    val optimizer = mock(classOf[Optimizer[DistributedGLMLossFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val regContext = mock(classOf[RegularizationContext])
    val optConfig = mock(classOf[FixedEffectOptimizationConfiguration])

    doReturn(Some(statesTracker)).when(optimizer).getStateTracker
    doReturn(regContext).when(optConfig).regularizationContext
    doReturn(RegularizationType.NONE).when(regContext).regularizationType

    val objective = DistributedGLMLossFunction(optConfig, LogisticLossFunction, treeAggregateDepth = 1)

    val optimizationProblem = new DistributedOptimizationProblem(
      optimizer,
      objective,
      samplerOption = None,
      glmConstructorMock,
      NoRegularizationContext,
      VarianceComputationType.FULL)

    // Produced by the reference implementation in R glm
    val expected = DenseVector(
      0.0007320271,
      0.3204454,
      0.05394657,
      0.0001520536,
      1.787598e-05,
      0.3898167,
      0.04483891,
      0.0001226556,
      0.2006968,
      0.05705076,
      0.1752335,
      0.08054471,
      0.01292064,
      10.37188)

    // From a prior optimization run
    val coefficients = DenseVector(
      -0.022306127,
      1.299914831,
      0.792316427,
      0.033470557,
      0.004679123,
      -0.459432925,
      0.294831754,
      -0.023566341,
      0.890054910,
      0.410533616,
      0.216417307,
      1.167698255,
      0.367261286,
      -8.303806435)
    val actual: Vector[Double] = optimizationProblem.computeVariances(input, coefficients).get

    VectorUtils.areAlmostEqual(actual, expected)
  }
}

object DistributedOptimizationProblemIntegTest {

  // No way to pass Mixin class type to Mockito, need to define a concrete class
  private class L2LossFunction(sc: SparkContext)
    extends DistributedSmoothedHingeLossFunction(treeAggregateDepth = 1)
      with L2RegularizationDiff
}
