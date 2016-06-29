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
package com.linkedin.photon.ml.supervised

import breeze.linalg.Vector
import com.linkedin.photon.ml.data.{LabeledPoint, SimpleObjectProvider}
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization.{NormalizationContext, NoNormalization}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.test.SparkTestUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.testng.Assert.assertTrue
import org.testng.Assert.fail
import org.testng.annotations.{DataProvider, Test}

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
 * For now, we focus only on the happy path tests. For now, we define those to be:
 * <ul>
 *   <li>LBFGS as the solver</li>
 *   <li>Each kind of GLA (LogisticRegression, LinearRegression, PoissonRegression when it's ready)</li>
 *   <li>Valid regularization weight (1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4)</li>
 *   <li>Summary is none</li>
 *   <li>Normalization is st. dev</li>
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
   * Enumerate valid sets of (description, generalized linear algorithm, data set) tuples.
   */
  private def getGeneralizedLinearOptimizationProblems: Array[(Object, Object, Object, Object)] = {
    val treeAggregateDepth = 1
    val enableStateTracker = true

    val lbfgsConfig = GLMOptimizationConfiguration(
      OptimizerConfig(OptimizerType.LBFGS, LBFGS.DEFAULT_MAX_ITER, LBFGS.DEFAULT_TOLERANCE, None),
      L2RegularizationContext)

    val tronConfig = GLMOptimizationConfiguration(
      OptimizerConfig(OptimizerType.TRON, TRON.DEFAULT_MAX_ITER, TRON.DEFAULT_TOLERANCE, None),
      L2RegularizationContext)

    Array(
      Tuple4("Linear regression, easy problem",
        LinearRegressionOptimizationProblem.buildOptimizationProblem(
          lbfgsConfig, treeAggregateDepth, enableStateTracker),
        drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_SAMPLES, BaseGLMIntegTest.NUMBER_OF_DIMENSIONS),
        new CompositeModelValidator[LinearRegressionModel](new PredictionFiniteValidator(),
          new MaximumDifferenceValidator[LinearRegressionModel](BaseGLMIntegTest.MAXIMUM_ERROR_MAGNITUDE))),

      Tuple4("Poisson regression, easy problem",
        PoissonRegressionOptimizationProblem.buildOptimizationProblem(
          lbfgsConfig, treeAggregateDepth, enableStateTracker),
        drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_SAMPLES, BaseGLMIntegTest.NUMBER_OF_DIMENSIONS),
        new CompositeModelValidator[PoissonRegressionModel](new PredictionFiniteValidator,
          new NonNegativePredictionValidator[PoissonRegressionModel]
        )
      ),

      Tuple4("Logistic regression, easy problem",
        LogisticRegressionOptimizationProblem.buildOptimizationProblem(
          tronConfig, treeAggregateDepth, enableStateTracker),
        drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_SAMPLES, BaseGLMIntegTest.NUMBER_OF_DIMENSIONS),
        new CompositeModelValidator[LogisticRegressionModel](new PredictionFiniteValidator(),
          new BinaryPredictionValidator[LogisticRegressionModel](),
          new BinaryClassifierAUCValidator[LogisticRegressionModel](BaseGLMIntegTest.MINIMUM_CLASSIFIER_AUCROC))),

      Tuple4("Linear SVM, easy problem",
        SmoothedHingeLossLinearSVMOptimizationProblem.buildOptimizationProblem(
          lbfgsConfig, treeAggregateDepth, enableStateTracker),
        drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_SAMPLES, BaseGLMIntegTest.NUMBER_OF_DIMENSIONS),
        new CompositeModelValidator[SmoothedHingeLossLinearSVMModel](new PredictionFiniteValidator(),
          new BinaryPredictionValidator[SmoothedHingeLossLinearSVMModel](),
          new BinaryClassifierAUCValidator[SmoothedHingeLossLinearSVMModel](BaseGLMIntegTest.MINIMUM_CLASSIFIER_AUCROC)))
    )
  }

  @DataProvider
  def generateHappyPathCases() : Array[Array[Object]] = {
    val toGenerate = getGeneralizedLinearOptimizationProblems

    toGenerate.map( x => {
      Array(
        s"Happy path [$x._1]",   // Description
        x._2,                    // Optimization Problem
        x._3,                    // data
        x._4                     // validator
      )
    })
  }

  @Test(dataProvider = "generateHappyPathCases")
  def checkHappyPath[GLM <: GeneralizedLinearModel, Function <: DiffFunction[LabeledPoint]](
      desc: String,
      optimizationProblem: GeneralizedLinearOptimizationProblem[GLM, Function],
      data: Iterator[(Double, Vector[Double])],
      validator: ModelValidator[GLM]) = {

    runGeneralizedLinearOptimizationProblemScenario(
      desc,
      optimizationProblem,
      List(1.0),
      None,
      NoNormalization,
      data,
      validator)
  }

  /**
   * Expectation is that there may be many clients of this method (e.g. several different tests that change bindings /
   * data providers to exercise different paths) -- hopefully the majority of tests can be constructed by creating
   * the right bindings.
   */
  def runGeneralizedLinearOptimizationProblemScenario[GLM <: GeneralizedLinearModel, Function <: DiffFunction[LabeledPoint]](
      desc: String,
      optimizationProblem: GeneralizedLinearOptimizationProblem[GLM, Function],
      lambdas: List[Double],
      summary: Option[BasicStatisticalSummary],
      norm: NormalizationContext,
      data: Iterator[(Double, Vector[Double])],
      validator: ModelValidator[GLM]) = sparkTest(desc) {

    val normalizationContextProvider = new SimpleObjectProvider[NormalizationContext](norm)

    // Step 1: generate our input RDD
    val trainingSet: RDD[LabeledPoint] = sc.parallelize(data.map( x => {
      new LabeledPoint(label = x._1, features = x._2)
    }).toList).repartition(BaseGLMIntegTest.NUM_PARTITIONS)

    // Step 2: actually run
    val models = lambdas.map { lambda =>
      val problem = optimizationProblem.updateObjective(normalizationContextProvider, lambda)
      problem.run(trainingSet, norm)
    }

    // Step 3: check convergence
    // TODO: Figure out if this test continues to make sense when we have multiple lambdas and, if not, how it should
    // TODO: be fixed.
    assertTrue(optimizationProblem.getStatesTracker.isDefined, "State tracking was enabled")
    OptimizerIntegTest.checkConvergence(optimizationProblem.getStatesTracker.get)

    // Step 4: validate the models
    models.foreach( m => {
      m.validateCoefficients()
      validator.validateModelPredictions(m, trainingSet)
    })
  }
}

/**
 * Mostly constants controlling this test
 */
object BaseGLMIntegTest {
  val NUM_PARTITIONS = 4
  val RANDOM_SEED:Int = 0
  // 10,000 samples would be good enough
  val NUMBER_OF_SAMPLES:Int = 10000
  // dimension of 10 would be sufficient to test the problem
  val NUMBER_OF_DIMENSIONS:Int = 10
  /** Minimum required AUROC */
  val MINIMUM_CLASSIFIER_AUCROC:Double = 0.95
  /** Maximum allowable magnitude difference between predictions and labels for regression problems
    * (this corresponds to 10 sigma, i.e. events that should occur at most once in the lifespan of our solar system)*/
  val MAXIMUM_ERROR_MAGNITUDE:Double = 10 * SparkTestUtils.INLIER_STANDARD_DEVIATION
}
