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
package com.linkedin.photon.ml.optimization

import java.util.Random

import breeze.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.{LabeledPoint, SimpleObjectProvider}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}

class PoissonRegressionOptimizationProblemTest extends SparkTestUtils{
  import CommonTestUtils._
  import PoissonRegressionOptimizationProblemTest._

  def generateUnweightedBenignLocalDataSet: List[LabeledPoint] = {
    drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      DIMENSIONS)
      .map { case (label, features) =>
        assertEquals(features.length, DIMENSIONS, "Samples should have expected lengths")

        new LabeledPoint(label, features)
      }
      .toList
  }

  def generateWeightedBenignLocalDataSet: List[LabeledPoint] = {
    val r: Random = new Random(WEIGHT_RANDOM_SEED)

    drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      DIMENSIONS)
      .map { case (label, features) =>
        val offset = 0D
        val weight = r.nextDouble() * WEIGHT_MAX
        assertEquals(features.length, DIMENSIONS, "Samples should have expected lengths")

        new LabeledPoint(label, features, offset, weight)
      }
      .toList
  }

  @DataProvider(parallel = true)
  def getDataAndWeights: Array[Array[Object]] = {
    val weightsToTest = Array(0.1, 1.0, 10.0, 100.0)
    val dataSet = generateWeightedBenignLocalDataSet

    weightsToTest.map( Array(_, dataSet).asInstanceOf[Array[Object]] )
  }

  @Test
  def testUpdateObjective(): Unit = {
    val problem = createProblem()
    val normalizationContext = new SimpleObjectProvider(mock(classOf[NormalizationContext]))
    val regularizationWeight = 1D

    assertNotEquals(problem.regularizationWeight, regularizationWeight)

    val updatedProblem = problem.updateObjective(normalizationContext, regularizationWeight)

    assertEquals(updatedProblem.regularizationWeight, regularizationWeight)
  }

  @Test
  def testInitializeZeroModel(): Unit = {
    val problem = createProblem()
    val zeroModel = problem.initializeZeroModel(DIMENSIONS)

    assertEquals(zeroModel.coefficients, Coefficients.initializeZeroCoefficients(DIMENSIONS))
  }

  @Test
  def testCreateModel(): Unit = {
    val problem = createProblem()
    val coefficients = generateDenseVector(DIMENSIONS)
    val model = problem.createModel(coefficients, None)

    assertEquals(model.coefficients.means, coefficients)
  }

  @Test
  def testComputeVariancesDisabled(): Unit = {
    val problem = createProblem()
    val input = mock(classOf[RDD[LabeledPoint]])
    val coefficients = generateDenseVector(DIMENSIONS)

    assertEquals(problem.computeVariances(input, coefficients), None)
  }

  @Test
  def testComputeVariancesSimple(): Unit = {
    val problem = createProblem(computeVariance = true)
    val input = generateUnweightedBenignLocalDataSet
    val coefficients = generateDenseVector(DIMENSIONS)

    // For Poisson regression, the second derivative of the loss function (with regard to z = X_i * B) is e^z.
    val hessianDiagonal: Vector[Double] = input.foldLeft(new DenseVector[Double](DIMENSIONS))
      { (diagonal: DenseVector[Double], datum: LabeledPoint) =>
        val features: Vector[Double] = datum.features
        val z: Double = datum.computeMargin(coefficients)
        val d2lossdz2 = math.exp(z)

        diagonal + (d2lossdz2 * features :* features)
      }
    // Simple estimate of the diagonal of the covariance matrix (instead of a full inverse).
    val expected: Vector[Double] = hessianDiagonal.map( v => 1D / (v + MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD) )

    val actual: Vector[Double] = problem.computeVariances(input, coefficients).get

    assertEquals(actual.length, DIMENSIONS)
    assertEquals(actual.length, expected.length)
    for (i <- 0 until DIMENSIONS) {
      assertEquals(actual(i), expected(i), MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)
    }
  }

  @Test(dataProvider = "getDataAndWeights")
  def testComputeVariancesComplex(regularizationWeight: Double, input: Iterable[LabeledPoint]): Unit = {
    val problem = createProblem(L2RegularizationContext, regularizationWeight, computeVariance = true)
    val coefficients = generateDenseVector(DIMENSIONS)

    // For linear regression, the second derivative of the loss function (with regard to z = X_i * B) is e^z.
    val hessianDiagonal: Vector[Double] = input.foldLeft(new DenseVector[Double](DIMENSIONS))
      { (diagonal: DenseVector[Double], datum: LabeledPoint) =>
        val features: Vector[Double] = datum.features
        val weight: Double = datum.weight
        val z: Double = datum.computeMargin(coefficients)
        val d2lossdz2 = math.exp(z)

        diagonal + (weight * d2lossdz2 * features :* features)
      }
    // Add the regularization to the Hessian. The second derivative of the L2 regularization term is the regularization
    // weight.
    val hessianDiagonalWithL2: Vector[Double] = hessianDiagonal + regularizationWeight
    // Simple estimate of the diagonal of the covariance matrix (instead of a full inverse).
    val expected: Vector[Double] = hessianDiagonalWithL2.map( v =>
      1D / (v + MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)
    )

    val actual: Vector[Double] = problem.computeVariances(input, coefficients).get

    assertEquals(actual.length, DIMENSIONS)
    assertEquals(actual.length, expected.length)
    for (i <- 0 until DIMENSIONS) {
      assertEquals(actual(i), expected(i), MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)
    }
  }
}

object PoissonRegressionOptimizationProblemTest {
  val DATA_RANDOM_SEED: Int = 7
  val WEIGHT_RANDOM_SEED: Int = 13
  val WEIGHT_MAX: Double = 10.0
  val DIMENSIONS: Int = 5
  val TRAINING_SAMPLES: Int = DIMENSIONS * DIMENSIONS

  def createProblem(
    regularizationContext: RegularizationContext = NoRegularizationContext,
    regularizationWeight: Double = 0D,
    computeVariance: Boolean = false): PoissonRegressionOptimizationProblem = {

    val config = new GLMOptimizationConfiguration(
      optimizerConfig = OptimizerConfig(OptimizerType.LBFGS, 100, 1E-10, None),
      regularizationContext = regularizationContext,
      regularizationWeight = regularizationWeight,
      downSamplingRate = 1D)
    val treeAggregateDepth = 1
    val isTrackingState = false

    PoissonRegressionOptimizationProblem.buildOptimizationProblem(
      config,
      treeAggregateDepth,
      isTrackingState,
      computeVariance)
  }
}
