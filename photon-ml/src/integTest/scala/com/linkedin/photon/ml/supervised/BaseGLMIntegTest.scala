package com.linkedin.photon.ml.supervised

import breeze.linalg.Vector
import com.linkedin.photon.ml.DataValidationType
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization.{NormalizationType, NormalizationContext, NoNormalization}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.classification.{SmoothedHingeLossLinearSVMModel, SmoothedHingeLossLinearSVMAlgorithm, LogisticRegressionAlgorithm, LogisticRegressionModel}
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearAlgorithm, GeneralizedLinearModel}
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionAlgorithm, LinearRegressionModel, PoissonRegressionAlgorithm, PoissonRegressionModel}
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
 *
 */
class BaseGLMIntegTest extends SparkTestUtils {
  /**
   * Enumerate valid sets of (description, generalized linear algorithm, data set) tuples.
   */
  private def getGeneralizedLinearAlgorithms() : Array[Tuple5[Object, Object, Object, Object, Object]] = {
    Array(
      Tuple5("Linear regression, easy problem",
        new LinearRegressionAlgorithm(),
        OptimizerConfig(OptimizerType.LBFGS, LBFGS.DEFAULT_MAX_ITER, LBFGS.DEFAULT_TOLERANCE, None),
        drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_SAMPLES, BaseGLMIntegTest.NUMBER_OF_DIMENSIONS),
        new CompositeModelValidator[LinearRegressionModel](new PredictionFiniteValidator(),
          new MaximumDifferenceValidator[LinearRegressionModel](BaseGLMIntegTest.MAXIMUM_ERROR_MAGNITUDE))),

      Tuple5("Poisson regression, easy problem",
        new PoissonRegressionAlgorithm,
        OptimizerConfig(OptimizerType.LBFGS, LBFGS.DEFAULT_MAX_ITER, LBFGS.DEFAULT_TOLERANCE, None),
        drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_SAMPLES, BaseGLMIntegTest.NUMBER_OF_DIMENSIONS),
          new CompositeModelValidator[PoissonRegressionModel](new PredictionFiniteValidator, new NonNegativePredictionValidator[PoissonRegressionModel])),

      Tuple5("Logistic regression, easy problem",
        new LogisticRegressionAlgorithm(),
        OptimizerConfig(OptimizerType.TRON, TRON.DEFAULT_MAX_ITER, TRON.DEFAULT_TOLERANCE, None),
        drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_SAMPLES, BaseGLMIntegTest.NUMBER_OF_DIMENSIONS),
        new CompositeModelValidator[LogisticRegressionModel](new PredictionFiniteValidator(),
          new BinaryPredictionValidator[LogisticRegressionModel](),
          new BinaryClassifierAUCValidator[LogisticRegressionModel](BaseGLMIntegTest.MINIMUM_CLASSIFIER_AUCROC))),

      Tuple5("Linear SVM, easy problem",
        new SmoothedHingeLossLinearSVMAlgorithm(),
        OptimizerConfig(OptimizerType.LBFGS, LBFGS.DEFAULT_MAX_ITER, LBFGS.DEFAULT_TOLERANCE, None),
        drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_SAMPLES, BaseGLMIntegTest.NUMBER_OF_DIMENSIONS),
        new CompositeModelValidator[SmoothedHingeLossLinearSVMModel](new PredictionFiniteValidator(),
          new BinaryPredictionValidator[SmoothedHingeLossLinearSVMModel](),
          new BinaryClassifierAUCValidator[SmoothedHingeLossLinearSVMModel](BaseGLMIntegTest.MINIMUM_CLASSIFIER_AUCROC)))
    )
  }

  @DataProvider
  def generateHappyPathCases() : Array[Array[Object]] = {
    val toGenerate = getGeneralizedLinearAlgorithms

    toGenerate.map( x => {
      Array(
        s"Happy path [$x._1]", // Description
        x._2,                                // Algorithm
        x._3,                                // Solver
        L2RegularizationContext,             // Regularization
        x._4,                                // data
        x._5                                 // validator
      )
    })
  }

  @Test(dataProvider = "generateHappyPathCases")
  def checkHappyPath[GLM <: GeneralizedLinearModel, Function <: DiffFunction[LabeledPoint]](
      desc: String,
      algorithm: GeneralizedLinearAlgorithm[GLM, Function],
      optimizerConfig: OptimizerConfig,
      reg: RegularizationContext,
      data: Iterator[(Double, Vector[Double])],
      validator: ModelValidator[GLM]) = {
    runGeneralizedLinearAlgorithmScenario(desc, algorithm, optimizerConfig, reg, List(1.0), None, NoNormalization, data, validator)
  }

  /**
   * Expectation is that there may be many clients of this method (e.g. several different tests that change bindings /
   * data providers to exercise different paths) -- hopefully the majority of tests can be constructed by creating
   * the right bindings.
   */
  def runGeneralizedLinearAlgorithmScenario[GLM <: GeneralizedLinearModel, Function <: DiffFunction[LabeledPoint]](
      desc: String,
      algorithm: GeneralizedLinearAlgorithm[GLM, Function],
      optimizerConfig: OptimizerConfig,
      reg: RegularizationContext,
      lambdas: List[Double],
      summary: Option[BasicStatisticalSummary],
      norm: NormalizationContext,
      data: Iterator[(Double, Vector[Double])],
      validator: ModelValidator[GLM]) = sparkTest(desc) {

    // Step 0: configure the algorithm
    algorithm.enableIntercept = true
    algorithm.isTrackingState = true
    algorithm.targetStorageLevel = StorageLevel.MEMORY_ONLY

    // Step 1: generate our input RDD
    val trainingSet: RDD[LabeledPoint] = sc.parallelize(data.map( x => { new LabeledPoint(label = x._1, features = x._2)}).toList).repartition(4)

    // Step 2: actually run
    val (models, optimizer) = algorithm.run(trainingSet, optimizerConfig, reg, lambdas, norm)

    // Step 3: check convergence
    // TODO: Figure out if this test continues to make sense when we have multiple lambdas and, if not, how it should
    // TODO: be fixed.
    assertTrue(None != optimizer.getStateTracker, "State tracking was enabled")
    OptimizerIntegTest.checkConvergence(optimizer.getStateTracker.get)

    // Step 4: validate the models
    models.foreach( m => {
      m.validateCoefficients
      validator.validateModelPredictions(m, trainingSet)
    })

    // Step 5: did it really use the optimizer I specified?
    optimizer match {
      case _: LBFGS[_] => if (optimizerConfig.optimizerType != OptimizerType.LBFGS) {
        fail("Wrong optimizer selected at runtime: $optimizerConfig.optimizerType (should be LBFGS)")
      }
      case _: TRON[_] => if (optimizerConfig.optimizerType != OptimizerType.TRON) {
        fail("Wrong optimizer selected at runtime: $optimizerConfig.optimizerType (should be TRON)")
      }
    }
  }
}

/**
 * Mostly constants controlling this test
 */
object BaseGLMIntegTest {
  val RANDOM_SEED:Int = 0
  val NUMBER_OF_SAMPLES:Int = 100000
  val NUMBER_OF_DIMENSIONS:Int = 100
  /** Minimum required AUROC */
  val MINIMUM_CLASSIFIER_AUCROC:Double = 0.95
  /** Maximum allowable magnitude difference between predictions and labels for regression problems
    * (this corresponds to 10 sigma, i.e. events that should occur at most once in the lifespan of our solar system)*/
  val MAXIMUM_ERROR_MAGNITUDE:Double = 10 * SparkTestUtils.INLIER_STANDARD_DEVIATION
}
