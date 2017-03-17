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

import breeze.linalg.{DenseVector, Vector}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
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
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}

/**
 *
 */
class DistributedOptimizationProblemTest extends SparkTestUtils {

  import CommonTestUtils._
  import DistributedOptimizationProblemTest._

  def glmConstructorMock(coefficients: Coefficients): GeneralizedLinearModel = mock(classOf[GeneralizedLinearModel])

  /**
   * Generate unweighted benign data sets for binary classification.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateBenignDataSetBinaryClassification: Seq[LabeledPoint] =
    drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      DIMENSIONS)
      .map { obj =>
        assertEquals(obj._2.length, DIMENSIONS, "Samples should have expected lengths")
        new LabeledPoint(label = obj._1, features = obj._2)
      }
      .toList

  /**
   * Generate weighted benign data sets for binary classification.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateWeightedBenignDataSetBinaryClassification: Seq[LabeledPoint] = {
    val r = new Random(WEIGHT_RANDOM_SEED)

    drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      DIMENSIONS)
      .map { obj =>
        assertEquals(obj._2.length, DIMENSIONS, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      }
      .toList
  }

  /**
   * Generate unweighted benign data sets for linear regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateBenignDataSetLinearRegression: Seq[LabeledPoint] =
    drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      DIMENSIONS)
      .map({ obj =>
        assertEquals(obj._2.length, DIMENSIONS, "Samples should have expected lengths")
        new LabeledPoint(label = obj._1, features = obj._2)
      })
      .toList

  /**
   * Generate weighted benign data sets for linear regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateWeightedBenignDataSetLinearRegression: Seq[LabeledPoint] = {
    val r = new Random(WEIGHT_RANDOM_SEED)

    drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      DIMENSIONS)
      .map { obj =>
        assertEquals(obj._2.length, DIMENSIONS, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      }
      .toList
  }

  /**
   * Generate unweighted benign data sets for Poisson regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateBenignDataSetPoissonRegression: Seq[LabeledPoint] =
    drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      DIMENSIONS)
      .map({ obj =>
        assertEquals(obj._2.length, DIMENSIONS, "Samples should have expected lengths")
        new LabeledPoint(label = obj._1, features = obj._2)
      })
      .toList

  /**
   * Generate weighted benign data sets for Poisson regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateWeightedBenignDataSetPoissonRegression: Seq[LabeledPoint] = {
    val r = new Random(WEIGHT_RANDOM_SEED)

    drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      DIMENSIONS)
      .map { obj =>
        assertEquals(obj._2.length, DIMENSIONS, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      }
      .toList
  }

  @DataProvider(parallel = true)
  def variancesSimpleInput(): Array[Array[Object]] =
    // Input data generation function, objective function, manual Hessian calculation function
    Array(
      Array(generateBenignDataSetBinaryClassification _, LogisticLossFunction, logisticHessianSum _),
      Array(generateBenignDataSetLinearRegression _, SquaredLossFunction, linearHessianSum _),
      Array(generateBenignDataSetPoissonRegression _, PoissonLossFunction, poissonHessianSum _))

  @DataProvider(parallel = true)
  def variancesComplexInput(): Array[Array[Object]] = {
    val regularizationWeights = Array[java.lang.Double](0.1, 1.0, 10.0, 100.0)

    // Regularization weight, input data generation function, objective function, manual Hessian calculation function
    regularizationWeights.flatMap { weight =>
      Array(
        Array[Object](
          weight,
          generateWeightedBenignDataSetBinaryClassification _,
          LogisticLossFunction,
          logisticHessianSum _),
        Array[Object](
          weight,
          generateWeightedBenignDataSetLinearRegression _,
          SquaredLossFunction,
          linearHessianSum _),
        Array[Object](
          weight,
          generateWeightedBenignDataSetPoissonRegression _,
          PoissonLossFunction,
          poissonHessianSum _))
    }
  }

  @Test
  def testUpdateRegularizationWeight(): Unit = sparkTest("checkEasyTestFunctionSparkNoInitialValue") {
    val initL1Weight = 1D
    val initL2Weight = 2D
    val finalL1Weight = 3D
    val finalL2Weight = 4D
    val finalElasticWeight = 5D
    val alpha = 0.75
    val elasticFinalL1Weight = finalElasticWeight * alpha
    val elasticFinalL2Weight = finalElasticWeight * (1 - alpha)

    val optimizerL1 = new OWLQN(initL1Weight, NORMALIZATION_MOCK)
    val optimizer = mock(classOf[Optimizer[DistributedSmoothedHingeLossFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val objectiveFunction = mock(classOf[DistributedSmoothedHingeLossFunction])
    val objectiveFunctionL2 = new L2LossFunction(sc)
    objectiveFunctionL2.l2RegularizationWeight = initL2Weight

    doReturn(Some(statesTracker)).when(optimizer).getStateTracker

    val l1Problem = new DistributedOptimizationProblem(
      optimizerL1,
      objectiveFunction,
      samplerOption = None,
      LogisticRegressionModel.apply,
      L1RegularizationContext,
      isComputingVariances = false)
    val l2Problem = new DistributedOptimizationProblem(
      optimizer,
      objectiveFunctionL2,
      samplerOption = None,
      LogisticRegressionModel.apply,
      L2RegularizationContext,
      isComputingVariances = false)
    val elasticProblem = new DistributedOptimizationProblem(
      optimizerL1,
      objectiveFunctionL2,
      samplerOption = None,
      LogisticRegressionModel.apply,
      ElasticNetRegularizationContext(alpha),
      isComputingVariances = false)

    // Check update to L1/L2 weights individually
    assertNotEquals(optimizerL1.l1RegularizationWeight, finalL1Weight, TOLERANCE)
    assertNotEquals(objectiveFunctionL2.l2RegularizationWeight, finalL2Weight, TOLERANCE)
    assertEquals(optimizerL1.l1RegularizationWeight, initL1Weight, TOLERANCE)
    assertEquals(objectiveFunctionL2.l2RegularizationWeight, initL2Weight, TOLERANCE)

    l1Problem.updateRegularizationWeight(finalL1Weight)
    l2Problem.updateRegularizationWeight(finalL2Weight)

    assertNotEquals(optimizerL1.l1RegularizationWeight, initL1Weight, TOLERANCE)
    assertNotEquals(objectiveFunctionL2.l2RegularizationWeight, initL2Weight, TOLERANCE)
    assertEquals(optimizerL1.l1RegularizationWeight, finalL1Weight, TOLERANCE)
    assertEquals(objectiveFunctionL2.l2RegularizationWeight, finalL2Weight, TOLERANCE)

    // Check updates to L1/L2 weights together
    optimizerL1.l1RegularizationWeight = initL1Weight
    objectiveFunctionL2.l2RegularizationWeight = initL2Weight

    assertNotEquals(optimizerL1.l1RegularizationWeight, elasticFinalL1Weight, TOLERANCE)
    assertNotEquals(objectiveFunctionL2.l2RegularizationWeight, elasticFinalL2Weight, TOLERANCE)
    assertEquals(optimizerL1.l1RegularizationWeight, initL1Weight, TOLERANCE)
    assertEquals(objectiveFunctionL2.l2RegularizationWeight, initL2Weight, TOLERANCE)

    elasticProblem.updateRegularizationWeight(finalElasticWeight)

    assertNotEquals(optimizerL1.l1RegularizationWeight, initL1Weight, TOLERANCE)
    assertNotEquals(objectiveFunctionL2.l2RegularizationWeight, initL2Weight, TOLERANCE)
    assertEquals(optimizerL1.l1RegularizationWeight, elasticFinalL1Weight, TOLERANCE)
    assertEquals(objectiveFunctionL2.l2RegularizationWeight, elasticFinalL2Weight, TOLERANCE)
  }

  @Test(dataProvider = "variancesSimpleInput")
  def testComputeVariancesSimple(
    dataGenerationFunction: () => Seq[LabeledPoint],
    lossFunction: PointwiseLossFunction,
    resultDirectDerivationFunction: (Vector[Double]) => (Vector[Double], LabeledPoint) => Vector[Double]): Unit =
    sparkTest("testComputeVariancesSimple") {
      val input = sc.parallelize(dataGenerationFunction())
      val coefficients = generateDenseVector(DIMENSIONS)

      val optimizer = mock(classOf[Optimizer[DistributedGLMLossFunction]])
      val statesTracker = mock(classOf[OptimizationStatesTracker])

      doReturn(Some(statesTracker)).when(optimizer).getStateTracker

      val configuration = GLMOptimizationConfiguration()
      val objective = DistributedGLMLossFunction(sc, configuration, treeAggregateDepth = 1)(lossFunction)

      val optimizationProblem = new DistributedOptimizationProblem(
        optimizer,
        objective,
        samplerOption = None,
        glmConstructorMock,
        NoRegularizationContext,
        isComputingVariances = true)

      val hessianDiagonal = input.treeAggregate(new DenseVector[Double](DIMENSIONS).asInstanceOf[Vector[Double]])(
        seqOp = resultDirectDerivationFunction(coefficients),
        combOp = (vector1: Vector[Double], vector2: Vector[Double]) => vector1 + vector2,
        depth = 1)
      // Simple estimate of the diagonal of the covariance matrix (instead of a full inverse).
      val expected = hessianDiagonal.map( v => 1D / (v + TOLERANCE))
      val actual: Vector[Double] = optimizationProblem.computeVariances(input, coefficients).get

      assertEquals(actual.length, DIMENSIONS)
      assertEquals(actual.length, expected.length)
      for (i <- 0 until DIMENSIONS) {
        assertEquals(actual(i), expected(i), TOLERANCE)
      }
    }

  @Test(dataProvider = "variancesComplexInput")
  def testComputeVariancesComplex(
    regularizationWeight: Double,
    dataGenerationFunction: () => Seq[LabeledPoint],
    lossFunction: PointwiseLossFunction,
    resultDirectDerivationFunction: (Vector[Double]) => (Vector[Double], LabeledPoint) => Vector[Double]): Unit =
    sparkTest("testComputeVariancesComplex") {
      val input = sc.parallelize(dataGenerationFunction())
      val coefficients = generateDenseVector(DIMENSIONS)

      val optimizer = mock(classOf[Optimizer[DistributedGLMLossFunction]])
      val statesTracker = mock(classOf[OptimizationStatesTracker])

      doReturn(Some(statesTracker)).when(optimizer).getStateTracker


      val configuration = GLMOptimizationConfiguration(
        regularizationContext = L2RegularizationContext,
        regularizationWeight = regularizationWeight)
      val objective = DistributedGLMLossFunction(sc, configuration, treeAggregateDepth = 1)(lossFunction)

      val optimizationProblem = new DistributedOptimizationProblem(
        optimizer,
        objective,
        samplerOption = None,
        glmConstructorMock,
        L2RegularizationContext,
        isComputingVariances = true)

      val hessianDiagonal = input.treeAggregate(new DenseVector[Double](DIMENSIONS).asInstanceOf[Vector[Double]])(
        seqOp = resultDirectDerivationFunction(coefficients),
        combOp = (vector1: Vector[Double], vector2: Vector[Double]) => vector1 + vector2,
        depth = 1)
      val hessianDiagonalWithL2 = hessianDiagonal + regularizationWeight
      // Simple estimate of the diagonal of the covariance matrix (instead of a full inverse).
      val expected = hessianDiagonalWithL2.map( v => 1D / (v + TOLERANCE))
      val actual: Vector[Double] = optimizationProblem.computeVariances(input, coefficients).get

      assertEquals(actual.length, DIMENSIONS)
      assertEquals(actual.length, expected.length)
      for (i <- 0 until DIMENSIONS) {
        assertEquals(actual(i), expected(i), TOLERANCE)
      }
    }

  @Test
  def testRun(): Unit = {
    val coefficients = new Coefficients(generateDenseVector(DIMENSIONS))

    val trainingData = mock(classOf[RDD[LabeledPoint]])
    val optimizer = mock(classOf[Optimizer[DistributedGLMLossFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val objectiveFunction = mock(classOf[DistributedGLMLossFunction])
    val initialModel = mock(classOf[GeneralizedLinearModel])

    doReturn(Some(statesTracker)).when(optimizer).getStateTracker

    val problem = new DistributedOptimizationProblem(
      optimizer,
      objectiveFunction,
      samplerOption = None,
      LogisticRegressionModel.apply,
      NoRegularizationContext,
      isComputingVariances = false)

    doReturn(NORMALIZATION_MOCK).when(optimizer).getNormalizationContext
    doReturn(coefficients).when(initialModel).coefficients
    doReturn((coefficients.means, None))
      .when(optimizer)
      .optimize(objectiveFunction, coefficients.means)(trainingData)
    val state = OptimizerState(coefficients.means, 0, generateDenseVector(DIMENSIONS), 0)
    doReturn(Array(state)).when(statesTracker).getTrackedStates

    val model = problem.run(trainingData, initialModel)

    assertEquals(coefficients, model.coefficients)
    assertEquals(problem.getModelTracker.get.length, 1)
  }
}

object DistributedOptimizationProblemTest {
  private val DATA_RANDOM_SEED: Int = 7
  private val WEIGHT_RANDOM_SEED = 100
  private val WEIGHT_RANDOM_MAX = 10
  private val DIMENSIONS: Int = 5
  private val TRAINING_SAMPLES: Int = DIMENSIONS * DIMENSIONS
  private val TOLERANCE = MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD
  private val NORMALIZATION = NoNormalization()
  private val NORMALIZATION_MOCK: Broadcast[NormalizationContext] = mock(classOf[Broadcast[NormalizationContext]])

  doReturn(NORMALIZATION).when(NORMALIZATION_MOCK).value

  /**
   *
   * @param coefficients
   * @param diagonal
   * @param datum
   * @return
   */
  def linearHessianSum
    (coefficients: Vector[Double])
    (diagonal: DenseVector[Double], datum: LabeledPoint): Vector[Double] = {

    // For linear regression, the second derivative of the loss function (with regard to z = X_i * B) is 1.
    val features: Vector[Double] = datum.features
    val weight: Double = datum.weight

    diagonal + (weight * features :* features)
  }

  /**
   *
   * @param coefficients
   * @param diagonal
   * @param datum
   * @return
   */
  def logisticHessianSum
    (coefficients: Vector[Double])
    (diagonal: DenseVector[Double], datum: LabeledPoint): Vector[Double] = {

    // For logistic regression, the second derivative of the loss function (with regard to z = X_i * B) is:
    //    sigmoid(z) * (1 - sigmoid(z))
    def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-z))

    val features: Vector[Double] = datum.features
    val weight: Double = datum.weight
    val z: Double = datum.computeMargin(coefficients)
    val sigm: Double = sigmoid(z)
    val d2lossdz2: Double = sigm * (1.0 - sigm)

    diagonal + (weight * d2lossdz2 * features :* features)
  }

  /**
   *
   * @param coefficients
   * @param diagonal
   * @param datum
   * @return
   */
  def poissonHessianSum
    (coefficients: Vector[Double])
    (diagonal: DenseVector[Double], datum: LabeledPoint): Vector[Double] = {

    // For Poisson regression, the second derivative of the loss function (with regard to z = X_i * B) is e^z.
    val features: Vector[Double] = datum.features
    val weight: Double = datum.weight
    val z: Double = datum.computeMargin(coefficients)
    val d2lossdz2 = math.exp(z)

    diagonal + (weight * d2lossdz2 * features :* features)
  }

  // No way to pass Mixin class type to Mockito, need to define a concrete class
  private class L2LossFunction(sc: SparkContext)
    extends DistributedSmoothedHingeLossFunction(sc, treeAggregateDepth = 1)
    with L2RegularizationDiff
}
