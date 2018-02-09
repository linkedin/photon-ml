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
package com.linkedin.photon.ml.function

import java.util.Random

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.SparkContext
import org.mockito.Mockito._
import org.testng.Assert.{assertEquals, assertTrue}
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.glm.{DistributedGLMLossFunction, LogisticLossFunction, PoissonLossFunction, SquaredLossFunction}
import com.linkedin.photon.ml.function.svm.DistributedSmoothedHingeLossFunction
import com.linkedin.photon.ml.normalization.NoNormalization
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.optimization.{L2RegularizationContext, NoRegularizationContext}
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.PhotonBroadcast

/**
 * Integration tests for [[DistributedObjectiveFunction]] to verify that the loss functions compute gradients & Hessians
 * accurately.
 */
class DistributedObjectiveFunctionIntegTest extends SparkTestUtils {

  import DistributedObjectiveFunctionIntegTest._

  private val twiceDiffTasks = Array(TaskType.LOGISTIC_REGRESSION, TaskType.LINEAR_REGRESSION, TaskType.POISSON_REGRESSION)
  private val diffTasks = twiceDiffTasks ++ Array(TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM)
  private val binaryClassificationDataSetGenerationFuncs = Array(
    generateBenignDataSetBinaryClassification _,
    generateWeightedBenignDataSetBinaryClassification _,
    generateOutlierDataSetBinaryClassification _,
    generateWeightedOutlierDataSetBinaryClassification _)
  private val linearRegressionDataSetGenerationFuncs = Array(
    generateBenignDataSetLinearRegression _,
    generateWeightedBenignDataSetLinearRegression _,
    generateOutlierDataSetLinearRegression _,
    generateWeightedOutlierDataSetLinearRegression _)
  private val poissonRegressionDataSetGenerationFuncs = Array(
    generateBenignDataSetPoissonRegression _,
    generateWeightedBenignDataSetPoissonRegression _,
    generateOutlierDataSetPoissonRegression _,
    generateWeightedOutlierDataSetPoissonRegression _)
  private val treeAggregateDepths = Array(1, 2)

  /**
   * Generate loss functions objects for classes implementing DiffFunction.
   *
   * @return Anonymous functions to generate the loss function and training data for the gradient tests
   */
  @DataProvider(parallel = true)
  def getDifferentiableFunctions: Array[Array[Object]] = diffTasks
    .flatMap {
      case TaskType.LOGISTIC_REGRESSION =>
        treeAggregateDepths.flatMap { treeAggDepth =>
          def lossFuncBuilder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, NO_REG_CONFIGURATION_MOCK, treeAggDepth)(LogisticLossFunction)
          def lossFuncWithL2Builder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, L2_REG_CONFIGURATION_MOCK, treeAggDepth)(LogisticLossFunction)

          binaryClassificationDataSetGenerationFuncs.flatMap { dataGenFunc =>
            Seq[(Object, Object)]((lossFuncBuilder, dataGenFunc), (lossFuncWithL2Builder, dataGenFunc))
          }
        }

      case TaskType.LINEAR_REGRESSION =>
        treeAggregateDepths.flatMap { treeAggDepth =>
          def lossFuncBuilder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, NO_REG_CONFIGURATION_MOCK, treeAggDepth)(SquaredLossFunction)
          def lossFuncWithL2Builder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, L2_REG_CONFIGURATION_MOCK, treeAggDepth)(SquaredLossFunction)

          linearRegressionDataSetGenerationFuncs.flatMap { dataGenFunc =>
            Seq[(Object, Object)]((lossFuncBuilder, dataGenFunc), (lossFuncWithL2Builder, dataGenFunc))
          }
        }

      case TaskType.POISSON_REGRESSION =>
        treeAggregateDepths.flatMap { treeAggDepth =>
          def lossFuncBuilder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, NO_REG_CONFIGURATION_MOCK, treeAggDepth)(PoissonLossFunction)
          def lossFuncWithL2Builder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, L2_REG_CONFIGURATION_MOCK, treeAggDepth)(PoissonLossFunction)

          poissonRegressionDataSetGenerationFuncs.flatMap { dataGenFunc =>
            Seq[(Object, Object)]((lossFuncBuilder, dataGenFunc), (lossFuncWithL2Builder, dataGenFunc))
          }
        }

      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        treeAggregateDepths.flatMap { treeAggDepth =>
          def lossFuncBuilder = (sc: SparkContext) => DistributedSmoothedHingeLossFunction(sc, NO_REG_CONFIGURATION_MOCK, treeAggDepth)
          def lossFuncWithL2Builder = (sc: SparkContext) => DistributedSmoothedHingeLossFunction(sc, L2_REG_CONFIGURATION_MOCK, treeAggDepth)

          binaryClassificationDataSetGenerationFuncs.flatMap { dataGenFunc =>
            Seq[(Object, Object)]((lossFuncBuilder, dataGenFunc), (lossFuncWithL2Builder, dataGenFunc))
          }
        }

      case other =>
        throw new IllegalArgumentException(s"Unrecognized task type: $other")
    }
    .map(pair => Array[Object](pair._1, pair._2))

  /**
   * Generate loss functions objects for classes implementing TwiceDiffFunction.
   *
   * @return Anonymous functions to generate the loss function and training data for the Hessian tests
   */
  @DataProvider(parallel = true)
  def getTwiceDifferentiableFunctions: Array[Array[Object]] = twiceDiffTasks
    .flatMap {
      case TaskType.LOGISTIC_REGRESSION =>
        treeAggregateDepths.flatMap { treeAggDepth =>
          def lossFuncBuilder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, NO_REG_CONFIGURATION_MOCK, treeAggDepth)(LogisticLossFunction)
          def lossFuncWithL2Builder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, L2_REG_CONFIGURATION_MOCK, treeAggDepth)(LogisticLossFunction)

          binaryClassificationDataSetGenerationFuncs.flatMap { dataGenFunc =>
            Seq((lossFuncBuilder, dataGenFunc), (lossFuncWithL2Builder, dataGenFunc))
          }
        }

      case TaskType.LINEAR_REGRESSION =>
        treeAggregateDepths.flatMap { treeAggDepth =>
          def lossFuncBuilder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, NO_REG_CONFIGURATION_MOCK, treeAggDepth)(SquaredLossFunction)
          def lossFuncWithL2Builder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, L2_REG_CONFIGURATION_MOCK, treeAggDepth)(SquaredLossFunction)

          linearRegressionDataSetGenerationFuncs.flatMap { dataGenFunc =>
            Seq((lossFuncBuilder, dataGenFunc), (lossFuncWithL2Builder, dataGenFunc))
          }
        }

      case TaskType.POISSON_REGRESSION =>
        treeAggregateDepths.flatMap { treeAggDepth =>
          def lossFuncBuilder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, NO_REG_CONFIGURATION_MOCK, treeAggDepth)(PoissonLossFunction)
          def lossFuncWithL2Builder = (sc: SparkContext) =>
            DistributedGLMLossFunction(sc, L2_REG_CONFIGURATION_MOCK, treeAggDepth)(PoissonLossFunction)

          poissonRegressionDataSetGenerationFuncs.flatMap { dataGenFunc =>
            Seq((lossFuncBuilder, dataGenFunc), (lossFuncWithL2Builder, dataGenFunc))
          }
        }

      case other =>
        throw new IllegalArgumentException(s"Unrecognized task type: $other")
    }
    .map(pair => Array[Object](pair._1, pair._2))

  //
  // Binary classification data set generation functions
  //

  /**
   * Generate unweighted benign data sets for binary classification.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateBenignDataSetBinaryClassification: Seq[LabeledPoint] =
    drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      PROBLEM_DIMENSION)
      .map { obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
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
      PROBLEM_DIMENSION)
      .map { obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      }
      .toList
  }

  /**
   * Generate unweighted outlier-dense data sets for binary classification.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateOutlierDataSetBinaryClassification: Seq[LabeledPoint] =
    drawBalancedSampleFromOutlierDenseFeaturesForBinaryClassifierLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      PROBLEM_DIMENSION)
      .map({ obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
        new LabeledPoint(label = obj._1, features = obj._2)
      })
      .toList

  /**
   * Generate weighted outlier-dense data sets for binary classification.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateWeightedOutlierDataSetBinaryClassification: Seq[LabeledPoint] = {
    val r = new Random(WEIGHT_RANDOM_SEED)

    drawBalancedSampleFromOutlierDenseFeaturesForBinaryClassifierLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      PROBLEM_DIMENSION)
      .map({ obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
        val weight = r.nextDouble() * WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      })
      .toList
  }

  //
  // Linear regression data set generation functions
  //

  /**
   * Generate unweighted benign data sets for linear regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateBenignDataSetLinearRegression: Seq[LabeledPoint] =
    drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      PROBLEM_DIMENSION)
      .map({ obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
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
      PROBLEM_DIMENSION)
      .map({ obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
        val weight = r.nextDouble() * WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      })
      .toList
  }

  /**
   * Generate unweighted outlier-dense data sets for linear regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateOutlierDataSetLinearRegression: Seq[LabeledPoint] =
    drawSampleFromOutlierDenseFeaturesForLinearRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      PROBLEM_DIMENSION)
      .map({ obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
        new LabeledPoint(label = obj._1, features = obj._2)
      })
      .toList

  /**
   * Generate weighted outlier-dense data sets for linear regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateWeightedOutlierDataSetLinearRegression: Seq[LabeledPoint] = {
    val r = new Random(WEIGHT_RANDOM_SEED)

    drawSampleFromOutlierDenseFeaturesForLinearRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      PROBLEM_DIMENSION)
      .map({ obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
        val weight = r.nextDouble() * WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      })
      .toList
  }

  //
  // Poisson regression data set generation functions
  //

  /**
   * Generate unweighted benign data sets for Poisson regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateBenignDataSetPoissonRegression: Seq[LabeledPoint] =
    drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      PROBLEM_DIMENSION)
      .map({ obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
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
      PROBLEM_DIMENSION)
      .map({ obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
        val weight = r.nextDouble() * WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      })
      .toList
  }

  /**
   * Generate unweighted outlier-dense data sets for Poisson regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateOutlierDataSetPoissonRegression: Seq[LabeledPoint] =
    drawSampleFromOutlierDenseFeaturesForPoissonRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      PROBLEM_DIMENSION)
      .map({ obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
        new LabeledPoint(label = obj._1, features = obj._2)
      })
      .toList

  /**
   * Generate weighted outlier-dense data sets for Poisson regression.
   *
   * @return A Seq of [[LabeledPoint]]
   */
  def generateWeightedOutlierDataSetPoissonRegression: Seq[LabeledPoint] = {
    val r = new Random(WEIGHT_RANDOM_SEED)

    drawSampleFromOutlierDenseFeaturesForPoissonRegressionLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      PROBLEM_DIMENSION)
      .map({ obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
        val weight = r.nextDouble() * WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, features = obj._2, weight = weight)
      })
      .toList
  }

  /**
   * Check a computed derivative value against a numerical estimate and throw an error if either the relative or
   * absolute error is too great.
   *
   * @param prefix An informative message about the derivative being checked
   * @param descriptor A description of the context for the numerical estimation
   * @param deltaBefore The function value at delta units before the computation
   * @param deltaAfter The function value at delta units after the computation
   * @param expected The computed value of the derivative
   * @param tolerance The relative & absolute error tolerance
   */
  def checkDerivativeError(
    prefix: String,
    descriptor: String,
    deltaBefore: Double,
    deltaAfter: Double,
    expected: Double,
    tolerance: Double): Unit = {

    assertTrue(java.lang.Double.isFinite(deltaBefore), s"Value before step [$deltaBefore] should be finite")
    assertTrue(java.lang.Double.isFinite(deltaAfter), s"Value after step [$deltaAfter] should be finite")
    assertTrue(java.lang.Double.isFinite(expected), s"Expected value [$expected] should be finite")

    val numericDerivative = (deltaAfter - deltaBefore) / (2.0 * DERIVATIVE_DELTA)
    val absoluteError = Math.abs(numericDerivative - expected)
    val relativeErrorNumerator = absoluteError
    val relativeErrorFactor = Math.min(Math.abs(numericDerivative), Math.abs(expected))
    val relativeErrorDenominator = if (relativeErrorFactor > 0) {
      relativeErrorFactor
    } else {
      1
    }
    val relativeError = relativeErrorNumerator / relativeErrorDenominator

    assert(
      relativeError < tolerance || absoluteError < tolerance,
      s"""$prefix
          |Computed and numerical differentiation estimates should be close.
          |NUMERICAL ESTIMATE: $numericDerivative ($descriptor)
          |COMPUTED: $expected
          |ABSOLUTE ERROR: $absoluteError
          |RELATIVE ERROR: $relativeError""".stripMargin)
  }

  /**
   * Verify that the gradient is accurate when computed via Spark.
   *
   * @note Rather than calling computeAt(...) to get both the objective and gradient in one shot, we use
   *       DiffFunction#value and DiffFunction#gradient instead. This is to ensure that we get some coverage of these
   *       functions which aren't used anywhere else. In the near term, we should decide if we want to keep those
   *       methods as part of the design or remove them, as they aren't used by any of the solvers
   * @param objectiveBuilder A builder function for the objective function object
   * @param dataGenerationFunction A builder function for the training dataset
   */
  @Test(dataProvider = "getDifferentiableFunctions", groups = Array[String]("ObjectiveFunctionTests", "testCore"))
  def checkGradientConsistentWithObjectiveSpark(
    objectiveBuilder: (SparkContext) => DistributedObjectiveFunction with DiffFunction,
    dataGenerationFunction: () => List[LabeledPoint]): Unit = sparkTest("checkGradientConsistentWithObjectiveSpark") {

    val objective = objectiveBuilder(sc)
    val trainingData = sc.parallelize(dataGenerationFunction()).repartition(NUM_PARTITIONS)
    val normalizationContextBroadcast = PhotonBroadcast(sc.broadcast(NORMALIZATION))
    val r = new Random(PARAMETER_RANDOM_SEED)

    for (iter <- 0 until SPARK_CONSISTENCY_CHECK_SAMPLES) {
      val initParam: Vector[Double] = DenseVector.fill[Double](PROBLEM_DIMENSION) {
        if (iter > 0) {
          r.nextDouble()
        } else {
          0
        }
      }
      val broadcastedInitParam = sc.broadcast(initParam)
      val computed = objective.gradient(trainingData, broadcastedInitParam, normalizationContextBroadcast)

      // Element-wise numerical differentiation to get the gradient
      for (idx <- 0 until PROBLEM_DIMENSION) {
        val before = initParam.copy
        before(idx) -= DERIVATIVE_DELTA
        val broadcastedBefore = sc.broadcast(before)

        val after = initParam.copy
        after(idx) += DERIVATIVE_DELTA
        val broadcastedAfter = sc.broadcast(after)

        val objBefore = objective.value(trainingData, broadcastedBefore, normalizationContextBroadcast)
        val objAfter = objective.value(trainingData, broadcastedAfter, normalizationContextBroadcast)

        broadcastedBefore.unpersist()
        broadcastedAfter.unpersist()

        checkDerivativeError(
          "Checking Gradient",
          s"f=[$objective / ${objective.getClass.getName}], iter=[$iter], idx=[$idx]",
          objBefore,
          objAfter,
          computed(idx),
          GRADIENT_TOLERANCE)
      }

      broadcastedInitParam.unpersist()
    }

    normalizationContextBroadcast.unpersist()
  }

  /**
   * Verify that the Hessian is accurate when computed via Spark.
   *
   * @param objectiveBuilder A builder function for the objective function object
   * @param dataGenerationFunction A builder function for the training dataset
   */
  @Test(dataProvider = "getTwiceDifferentiableFunctions", groups = Array[String]("ObjectiveFunctionTests", "testCore"))
  def checkHessianConsistentWithObjectiveSpark(
    objectiveBuilder: (SparkContext) => DistributedObjectiveFunction with TwiceDiffFunction,
    dataGenerationFunction: () => List[LabeledPoint]): Unit = sparkTest("checkHessianConsistentWithObjectiveSpark") {

    val objective = objectiveBuilder(sc)
    val trainingData = sc.parallelize(dataGenerationFunction()).repartition(NUM_PARTITIONS)
    val normalizationContextBroadcast = PhotonBroadcast(sc.broadcast(NORMALIZATION))
    val r = new Random(PARAMETER_RANDOM_SEED)

    for (iter <- 0 until SPARK_CONSISTENCY_CHECK_SAMPLES) {
      val initParam: Vector[Double] = DenseVector.fill[Double](PROBLEM_DIMENSION) {
        if (iter > 0) {
          r.nextDouble()
        } else {
          0
        }
      }
      val broadcastedInitParam = sc.broadcast(initParam)

      // Loop over basis vectors. This will give us H*e_i = H(:,i) (so one column of H at a time)
      for (basis <- 0 until PROBLEM_DIMENSION) {
        val basisVector: Vector[Double] = new SparseVector[Double](
          Array[Int](basis),
          Array[Double](1.0),
          1,
          PROBLEM_DIMENSION)
        val broadcastedBasis = sc.broadcast(basisVector)
        val hessianVector = objective.hessianVector(
          trainingData,
          broadcastedInitParam,
          broadcastedBasis,
          normalizationContextBroadcast)

        // Element-wise numerical differentiation to get the Hessian
        for (idx <- 0 until PROBLEM_DIMENSION) {
          val before = initParam.copy
          before(idx) -= DERIVATIVE_DELTA
          val broadcastedBefore = sc.broadcast(before)

          val after = initParam.copy
          after(idx) += DERIVATIVE_DELTA
          val broadcastedAfter = sc.broadcast(after)

          val gradBefore = objective.gradient(trainingData, broadcastedBefore, normalizationContextBroadcast)
          val gradAfter = objective.gradient(trainingData, broadcastedAfter, normalizationContextBroadcast)

          broadcastedBefore.unpersist()
          broadcastedAfter.unpersist()

          checkDerivativeError(
            "Checking Hessian",
            s"f=[$objective / ${objective.getClass.getName}], iter=[$iter], basis=[$basis], idx=[$idx], Hessian=[$hessianVector]",
            gradBefore(basis),
            gradAfter(basis),
            hessianVector(idx),
            HESSIAN_TOLERANCE)
        }

        broadcastedBasis.unpersist()
      }

      broadcastedInitParam.unpersist()
    }

    normalizationContextBroadcast.unpersist()
  }
}

object DistributedObjectiveFunctionIntegTest {
  private val SPARK_CONSISTENCY_CHECK_SAMPLES = 5
  private val NUM_PARTITIONS = 4
  private val PROBLEM_DIMENSION = 5
  private val NORMALIZATION = NoNormalization()
  private val L2_REG_CONFIGURATION_MOCK = mock(classOf[GLMOptimizationConfiguration])
  private val NO_REG_CONFIGURATION_MOCK = mock(classOf[GLMOptimizationConfiguration])
  private val REGULARIZATION_WEIGHT = 100.0
  private val DERIVATIVE_DELTA = 1e-6
  private val GRADIENT_TOLERANCE = 1e-3
  private val HESSIAN_TOLERANCE = 1e-3
  private val DATA_RANDOM_SEED = 0
  private val PARAMETER_RANDOM_SEED = 500
  private val WEIGHT_RANDOM_SEED = 100
  private val WEIGHT_RANDOM_MAX = 10
  private val TRAINING_SAMPLES = PROBLEM_DIMENSION * PROBLEM_DIMENSION

  doReturn(L2RegularizationContext).when(L2_REG_CONFIGURATION_MOCK).regularizationContext
  doReturn(REGULARIZATION_WEIGHT).when(L2_REG_CONFIGURATION_MOCK).regularizationWeight
  doReturn(NoRegularizationContext).when(NO_REG_CONFIGURATION_MOCK).regularizationContext
}
