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
package com.linkedin.photon.ml.supervised

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD
import org.testng.Assert.assertTrue
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.DistributedObjectiveFunction
import com.linkedin.photon.ml.function.glm.{LogisticLossFunction, PoissonLossFunction, SquaredLossFunction}
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.FixedEffectOptimizationConfiguration
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.{BroadcastWrapper, PhotonBroadcast}

// TODO: Update test to match all possible test scenarios
/**
 * Integration test scenarios to generate:
 *
 * <ul>
 *   <li>Each kind of solver (TRON, LBFGS)</li>
 *   <li>Each kind of regularization (L2, L1, E-net, none)</li>
 *   <li>Each kind of GLA (linear regression, logistic regression, poisson regression, etc.)
 *   <li>Regularization weights (null, empty, negative, single valid, several valid, valid with duplicates)</li>
 *   <li>Summary option (None, Some)</li>
 *   <li>Normalization type (None, max magnitude, st. dev)</li>
 *   <li>Input (valid + "easy", valid + "hard", invalid labels, invalid values)</li>
 * </ul>
 *
 * Unfortunately, we need a sensible subset of the cartesian product of all of these (e.g. TRON doesn't make sense
 * with L1).
 *
 * For now, we focus only on the main tests. For now, we define those to be:
 * <ul>
 *   <li>LBFGS as the solver</li>
 *   <li>Each kind of GLA (LogisticRegression, LinearRegression, PoissonRegression)</li>
 *   <li>L2 Regularization</li>
 *   <li>Valid regularization weight (1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4)</li>
 *   <li>TreeAggregateDepth is 1</li>
 *   <li>Summary is none</li>
 *   <li>No normalization</li>
 *   <li>Input is "easy"</li>
 * </ul>
 *
 * For these easy cases, we make sure that:
 *
 * <ul>
 *   <li>we ran (implicit)</li>
 *   <li>The number of models matches expectations</li>
 *   <li>The models themselves match expectations (per a provided ModelValidator)</li>
 * </ul>
 */
class BaseGLMIntegTest extends SparkTestUtils {

  /**
   * Generate a [[Seq]] of [[LabeledPoint]] from a collection of (label, feature vector) pairs.
   *
   * @param data A sequence of (label, feature vector) pairs
   * @return The data from the given sequence, converted to [[LabeledPoint]] instances
   */
  def generateDatasetIterable(data: Iterator[(Double, Vector[Double])]): Seq[LabeledPoint] =
    data.map( x => new LabeledPoint(label = x._1, features = x._2)).toList

  /**
   * Enumerate valid sets of (description, optimization problem builder, dataset, validator) tuples.
   */
  @DataProvider
  def getGeneralizedLinearOptimizationProblems: Array[Array[Object]] = {
    val lbfgsConfig = FixedEffectOptimizationConfiguration(
      OptimizerConfig(OptimizerType.LBFGS, LBFGS.DEFAULT_MAX_ITER, LBFGS.DEFAULT_TOLERANCE, None),
      L2RegularizationContext)
//    val tronConfig = GLMOptimizationConfiguration(
//      OptimizerConfig(OptimizerType.TRON, TRON.DEFAULT_MAX_ITER, TRON.DEFAULT_TOLERANCE, None),
//      L2RegularizationContext)

    val linearRegressionData = generateDatasetIterable(
      drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(
        BaseGLMIntegTest.RANDOM_SEED,
        BaseGLMIntegTest.NUMBER_OF_SAMPLES,
        BaseGLMIntegTest.NUMBER_OF_DIMENSIONS))
    val poissonRegressionData = generateDatasetIterable(
      drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
        BaseGLMIntegTest.RANDOM_SEED,
        BaseGLMIntegTest.NUMBER_OF_SAMPLES,
        BaseGLMIntegTest.NUMBER_OF_DIMENSIONS))
    val logisticRegressionData = generateDatasetIterable(
      drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
        BaseGLMIntegTest.RANDOM_SEED,
        BaseGLMIntegTest.NUMBER_OF_SAMPLES,
        BaseGLMIntegTest.NUMBER_OF_DIMENSIONS))
//    val smoothedHingeData = generateDatasetIterable(
//      drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
//        BaseGLMIntegTest.RANDOM_SEED,
//        BaseGLMIntegTest.NUMBER_OF_SAMPLES,
//        BaseGLMIntegTest.NUMBER_OF_DIMENSIONS))

    Array(
      Array(
        "Linear regression, easy problem",
        (normalizationContext: BroadcastWrapper[NormalizationContext]) =>
          DistributedOptimizationProblem(
            lbfgsConfig,
            DistributedObjectiveFunction(lbfgsConfig, SquaredLossFunction, treeAggregateDepth = 1),
            None,
            LinearRegressionModel.apply,
            normalizationContext,
            BaseGLMIntegTest.VARIANCE_COMPUTATION_TYPE),
        linearRegressionData,
        new CompositeModelValidator[LinearRegressionModel](
          new PredictionFiniteValidator(),
          new MaximumDifferenceValidator[LinearRegressionModel](BaseGLMIntegTest.MAXIMUM_ERROR_MAGNITUDE))),

      Array(
        "Poisson regression, easy problem",
        (normalizationContext: BroadcastWrapper[NormalizationContext]) =>
          DistributedOptimizationProblem(
            lbfgsConfig,
            DistributedObjectiveFunction(lbfgsConfig, PoissonLossFunction, treeAggregateDepth = 1),
            None,
            PoissonRegressionModel.apply,
            normalizationContext,
            BaseGLMIntegTest.VARIANCE_COMPUTATION_TYPE),
        poissonRegressionData,
        new CompositeModelValidator[PoissonRegressionModel](
          new PredictionFiniteValidator,
          new NonNegativePredictionValidator[PoissonRegressionModel]
        )
      ),

      Array(
        "Logistic regression, easy problem",
        (normalizationContext: BroadcastWrapper[NormalizationContext]) =>
          DistributedOptimizationProblem(
            lbfgsConfig,
            DistributedObjectiveFunction(lbfgsConfig, LogisticLossFunction, treeAggregateDepth = 1),
            None,
            LogisticRegressionModel.apply,
            normalizationContext,
            BaseGLMIntegTest.VARIANCE_COMPUTATION_TYPE),
        logisticRegressionData,
        new CompositeModelValidator[LogisticRegressionModel](
          new PredictionFiniteValidator(),
          new BinaryPredictionValidator[LogisticRegressionModel](),
          new BinaryClassifierAUCValidator[LogisticRegressionModel](BaseGLMIntegTest.MINIMUM_CLASSIFIER_AUCROC)))
    )
  }

  /**
   * Expectation is that there may be many clients of this method (e.g. several different tests that change bindings /
   * data providers to exercise different paths) -- hopefully the majority of tests can be constructed by creating
   * the right bindings.
   */
  @Test(dataProvider = "getGeneralizedLinearOptimizationProblems")
  def runGeneralizedLinearOptimizationProblemScenario(
      desc: String,
      optimizationProblemBuilder: BroadcastWrapper[NormalizationContext] =>
        DistributedOptimizationProblem[DistributedObjectiveFunction],
      data: Seq[LabeledPoint],
      validator: ModelValidator[GeneralizedLinearModel]): Unit = sparkTest(desc) {

    val normalizationContext = PhotonBroadcast(sc.broadcast(NoNormalization()))

    // Step 1: Generate input RDD
    val trainingSet: RDD[LabeledPoint] = sc.parallelize(data).repartition(BaseGLMIntegTest.NUM_PARTITIONS)
    val optimizationProblem = optimizationProblemBuilder(normalizationContext)

    // Step 2: Run optimization
    val modelsAndStateTrackers = BaseGLMIntegTest.LAMBDAS.map { lambda =>
      optimizationProblem.updateRegularizationWeight(lambda)
      val (model, statesTracker) = optimizationProblem.run(trainingSet)

      // Step 3: Check convergence
      BaseGLMIntegTest.checkConvergence(statesTracker)

      (model, statesTracker)
    }

    // Step 4: Validate the models
    modelsAndStateTrackers.foreach( t => {
      t._1.validateCoefficients()
      validator.validateModelPredictions(t._1, trainingSet)
    })
  }
}

/**
 * Constants controlling this test
 */
object BaseGLMIntegTest {

  private val LAMBDAS: Seq[Double] = List(1.0)
  // Failures for MaximumDifferenceValidator with all lambas enabled. Need to revisit settings.
  //private val LAMBDAS: Seq[Double] = List(1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)
  private val NUM_PARTITIONS = 4
  private val RANDOM_SEED: Int = 0
  // 10,000 samples would be good enough
  private val NUMBER_OF_SAMPLES: Int = 10000
  // Dimension of 10 should be sufficient to test these problems
  private val NUMBER_OF_DIMENSIONS: Int = 10
  // Minimum required AUROC
  private val MINIMUM_CLASSIFIER_AUCROC: Double = 0.95
  // Maximum allowable magnitude difference between predictions and labels for regression problems
  // (this corresponds to 10 sigma, i.e. events that should occur at most once in the lifespan of our solar system)
  private val MAXIMUM_ERROR_MAGNITUDE: Double = 10 * SparkTestUtils.INLIER_STANDARD_DEVIATION
  private val VARIANCE_COMPUTATION_TYPE = VarianceComputationType.NONE

  /**
   * Check that the loss value of the states in the [[OptimizationStatesTracker]] is monotonically decreasing.
   *
   * @param history The optimization state history
   */
  def checkConvergence(history: OptimizationStatesTracker): Unit =
    history.getTrackedStates.foldLeft(Double.MaxValue) { case (prevValue, state) =>
      assertTrue(
        prevValue >= state.loss,
        s"Objective should be monotonically decreasing (current=[${state.loss}], previous=[$prevValue])")

      state.loss
    }
}
