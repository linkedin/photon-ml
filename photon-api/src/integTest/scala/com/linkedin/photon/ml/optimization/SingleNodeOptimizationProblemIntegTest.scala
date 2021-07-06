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

import java.util.Random

import scala.io.Source

import breeze.linalg.{DenseMatrix, DenseVector, Vector, diag, pinv}
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.function.glm._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.optimization.game.FixedEffectOptimizationConfiguration
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}
import com.linkedin.photon.ml.util.VectorUtils

/**
 * Integration tests for [[SingleNodeOptimizationProblem]].
 */
class SingleNodeOptimizationProblemIntegTest extends SparkTestUtils {

  import CommonTestUtils._

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

    val linearData = generateWeightedBenignDatasetLinearRegression
    val logisticData = generateWeightedBenignDatasetBinaryClassification
    val poissonData = generateWeightedBenignDatasetPoissonRegression

    // Regularization weight, input data generation function, objective function, manual Hessian calculation function
    regularizationWeights.flatMap { weight =>
      Array(
        Array[Any](
          weight,
          logisticData,
          LogisticLossFunction,
          OptimizationProblemIntegTestUtils.logisticDzzLoss _),
        Array[Any](
          weight,
          linearData,
          SquaredLossFunction,
          OptimizationProblemIntegTestUtils.linearDzzLoss _),
        Array[Any](
          weight,
          poissonData,
          PoissonLossFunction,
          OptimizationProblemIntegTestUtils.poissonDzzLoss _))
    }
  }

  /**
   * Test simple coefficient variance computation for weighted data points, with regularization.
   *
   * @param regularizationWeight Regularization weight
   * @param inputData Input test data
   * @param lossFunction Loss function for optimization
   * @param DzzLossFunction Function to compute coefficient Hessian directly
   */
  @Test(dataProvider = "varianceInput")
  def testComputeVariancesSimple(
      regularizationWeight: Double,
      inputData: Seq[LabeledPoint],
      lossFunction: PointwiseLossFunction,
      DzzLossFunction: Vector[Double] => (LabeledPoint => Double)): Unit = {

    val coefficients = generateDenseVector(OptimizationProblemIntegTestUtils.DIMENSIONS)

    val optimizer = mock(classOf[Optimizer[SingleNodeObjectiveFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val regContext = mock(classOf[RegularizationContext])
    val optConfig = mock(classOf[FixedEffectOptimizationConfiguration])

    doReturn(statesTracker).when(optimizer).getStateTracker
    doReturn(regContext).when(optConfig).regularizationContext
    doReturn(regularizationWeight).when(optConfig).regularizationWeight
    doReturn(RegularizationType.L2).when(regContext).regularizationType
    doReturn(regularizationWeight).when(regContext).getL2RegularizationWeight(regularizationWeight)

    val objective = SingleNodeObjectiveFunction(optConfig, lossFunction)

    val optimizationProblem = new SingleNodeOptimizationProblem(
      optimizer,
      objective,
      glmConstructorMock,
      VarianceComputationType.SIMPLE)

    val hessianDiagonal = inputData.aggregate(DenseVector.zeros[Double](OptimizationProblemIntegTestUtils.DIMENSIONS))(
      seqop = (vector: DenseVector[Double], datum: LabeledPoint) => {
        diag(OptimizationProblemIntegTestUtils.hessianSum(DzzLossFunction(coefficients))(diag(vector), datum))
      },
      combop = (vector1: DenseVector[Double], vector2: DenseVector[Double]) => vector1 + vector2)
    // Simple estimate of the diagonal of the covariance matrix (instead of a full inverse).
    val expected = (hessianDiagonal + regularizationWeight).map( v => 1D / (v + MathConst.EPSILON))
    val actual: Vector[Double] = optimizationProblem.computeVariances(inputData, coefficients).get

    assertTrue(VectorUtils.areAlmostEqual(actual, expected))
  }

  /**
   * Test full coefficient variance computation for weighted data points, with regularization.
   *
   * @param regularizationWeight Regularization weight
   * @param inputData Input test data
   * @param lossFunction Loss function for optimization
   * @param DzzLossFunction Function to compute coefficient Hessian directly
   */
  @Test(dataProvider = "varianceInput")
  def testComputeVariancesFull(
      regularizationWeight: Double,
      inputData: Seq[LabeledPoint],
      lossFunction: PointwiseLossFunction,
      DzzLossFunction: Vector[Double] => (LabeledPoint => Double)): Unit = {

    val dimensions = OptimizationProblemIntegTestUtils.DIMENSIONS
    val coefficients = generateDenseVector(dimensions)

    val optimizer = mock(classOf[Optimizer[SingleNodeObjectiveFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val regContext = mock(classOf[RegularizationContext])
    val optConfig = mock(classOf[FixedEffectOptimizationConfiguration])

    doReturn(statesTracker).when(optimizer).getStateTracker
    doReturn(regContext).when(optConfig).regularizationContext
    doReturn(regularizationWeight).when(optConfig).regularizationWeight
    doReturn(RegularizationType.L2).when(regContext).regularizationType
    doReturn(regularizationWeight).when(regContext).getL2RegularizationWeight(regularizationWeight)

    val objective = SingleNodeObjectiveFunction(optConfig, lossFunction)

    val optimizationProblem = new SingleNodeOptimizationProblem(
      optimizer,
      objective,
      glmConstructorMock,
      VarianceComputationType.FULL)

    val hessianMatrix = inputData.aggregate(
      DenseMatrix.zeros[Double](dimensions, dimensions))(
      seqop = OptimizationProblemIntegTestUtils.hessianSum(DzzLossFunction(coefficients)),
      combop = (matrix1: DenseMatrix[Double], matrix2: DenseMatrix[Double]) => matrix1 + matrix2)
    // Simple estimate of the diagonal of the covariance matrix (instead of a full inverse).
    val expected = diag(pinv(hessianMatrix + (DenseMatrix.eye[Double](dimensions) * regularizationWeight)))
    val actual: Vector[Double] = optimizationProblem.computeVariances(inputData, coefficients).get

    assertTrue(VectorUtils.areAlmostEqual(actual, expected))
  }

  /**
   * Test the variance computation against a reference implementation in R glm.
   */
  @Test
  def testComputeVariancesAgainstReference(): Unit = {

    // Read the "heart disease" dataset from libSVM format
    val input = Source
      .fromFile(getClass.getClassLoader.getResource("DriverIntegTest/input/heart.txt").toURI)
      .getLines()
      .map { x =>
        val y = x.split(" ")
        val label = y(0).toDouble / 2 + 0.5
        val features = y.drop(1).map(z => z.split(":")(1).toDouble) :+ 1.0

        new LabeledPoint(label, DenseVector(features))
      }

    val optimizer = mock(classOf[Optimizer[SingleNodeObjectiveFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val regContext = mock(classOf[RegularizationContext])
    val optConfig = mock(classOf[FixedEffectOptimizationConfiguration])

    doReturn(statesTracker).when(optimizer).getStateTracker
    doReturn(regContext).when(optConfig).regularizationContext
    doReturn(RegularizationType.NONE).when(regContext).regularizationType

    val objective = SingleNodeObjectiveFunction(optConfig, LogisticLossFunction)

    val optimizationProblem = new SingleNodeOptimizationProblem(
      optimizer,
      objective,
      glmConstructorMock,
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
    val actual: Vector[Double] = optimizationProblem.computeVariances(input.toIterable, coefficients).get

    VectorUtils.areAlmostEqual(actual, expected)
  }
}
