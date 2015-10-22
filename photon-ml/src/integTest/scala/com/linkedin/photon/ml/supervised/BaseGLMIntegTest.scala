package com.linkedin.photon.ml.supervised

import breeze.linalg.Vector
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.LinearRegressionModel
import com.linkedin.photon.ml.supervised.regression.PoissonRegressionModel
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, LogisticRegressionAlgorithm}
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, GeneralizedLinearAlgorithm}
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, LinearRegressionAlgorithm, PoissonRegressionModel, PoissonRegressionAlgorithm}
import com.linkedin.photon.ml.test.SparkTestUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.testng.Assert.assertTrue
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
  private def getGeneralizedLinearAlgorithms() : Array[Tuple4[Object, Object, Object, Object]] = {
    Array(
      Tuple4("Linear regression, easy problem",
        new LinearRegressionAlgorithm(),
        drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_SAMPLES, BaseGLMIntegTest.NUMBER_OF_DIMENSIONS),
        new CompositeModelValidator[LinearRegressionModel](new PredictionFiniteValidator(),
          new MaximumDifferenceValidator[LinearRegressionModel](BaseGLMIntegTest.MAXIMUM_ERROR_MAGNITUDE))),
      Tuple4("Logistic regression, easy problem",
        new LogisticRegressionAlgorithm(),
        drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_SAMPLES, BaseGLMIntegTest.NUMBER_OF_DIMENSIONS),
        new CompositeModelValidator[LogisticRegressionModel](new PredictionFiniteValidator(),
          new BinaryPredictionValidator[LogisticRegressionModel](),
          new BinaryClassifierAUCValidator[LogisticRegressionModel](BaseGLMIntegTest.MINIMUM_CLASSIFIER_AUCROC)))
    )
  }

  @DataProvider
  def generateHappyPathCases() : Array[Array[Object]] = {
    val toGenerate = getGeneralizedLinearAlgorithms

    toGenerate.map( x => {
      Array(
        s"Happy path [$x._1]", // Description
        x._2,                                // Algorithm
        new LBFGS(),                         // Solver
        L2RegularizationContext,             // Regularization
        x._3,                                // data
        x._4                                 // validator
      )
    }).toArray
  }

  @Test(dataProvider = "generateHappyPathCases")
  def checkHappyPath[GLM <: GeneralizedLinearModel, Function <: DiffFunction[LabeledPoint]](
                     desc:String,
                     algorithm:GeneralizedLinearAlgorithm[GLM, Function],
                     solver:Optimizer[LabeledPoint, Function],
                     reg:RegularizationContext,
                     data:Iterator[(Double, Vector[Double])],
                     validator:ModelValidator[GLM]) = {
    runGeneralizedLinearAlgorithmScenario(desc, algorithm , solver, reg, List(1.0), None, NormalizationType.NO_SCALING, data, validator)
  }

  @Test(dataProvider = "generateHappyPathCases", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def checkInvalidOffset[GLM <: GeneralizedLinearModel, Function <: DiffFunction[LabeledPoint]](
                                                                                             desc:String,
                                                                                             algorithm:GeneralizedLinearAlgorithm[GLM, Function],
                                                                                             solver:Optimizer[LabeledPoint, Function],
                                                                                             reg:RegularizationContext,
                                                                                             data:Iterator[(Double, Vector[Double])],
                                                                                             validator:ModelValidator[GLM]) = {
    runInvalidOffsetScenario(desc, algorithm , solver, reg, List(1.0), None, NormalizationType.NO_SCALING, data, validator)
  }

  /**
   * Enumerate sets of (description, generalized linear algorithm, invalid data set) tuples.
   */
  private def getGeneralizedLinearAlgorithmsInvalidFeatures() : Array[Tuple4[Object, Object, Object, Object]] = {
    return Array(
      Tuple4("Linear regression, NaN/Inf features",
        new LinearRegressionAlgorithm(),
        drawSampleFromInvalidDenseFeaturesForLinearRegressionLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLES, BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLE_DIMENSIONS),
        new PredictionFiniteValidator()),
      Tuple4("Poisson regression, NaN/Inf features",
        new PoissonRegressionAlgorithm(),
        drawSampleFromInvalidDenseFeaturesForPoissonRegressionLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLES, BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLE_DIMENSIONS),
        new PredictionFiniteValidator()),
      Tuple4("Logistic regression, NaN/Inf features",
        new LogisticRegressionAlgorithm(),
        drawBalancedSampleFromInvalidDenseFeaturesForBinaryClassifierLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLES, BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLE_DIMENSIONS),
        new CompositeModelValidator[LogisticRegressionModel](new PredictionFiniteValidator(), new BinaryPredictionValidator[LogisticRegressionModel]())))
  }

  @DataProvider
  def generateInvalidFeatureCases() : Array[Array[Object]] = {
    val toGenerate = getGeneralizedLinearAlgorithmsInvalidFeatures

    toGenerate.map( x => {
      Array(
        s"Invalid features [$x._1]", // Description
        x._2,                        // Algorithm
        new LBFGS(),                 // Solver
        L2RegularizationContext,     // Regularization
        x._3,                        // data
        x._4                         // validator
      )
    }).toArray
  }

  @Test(dataProvider = "generateInvalidFeatureCases", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def checkInvalidFeatures[GLM <: GeneralizedLinearModel, Function <: DiffFunction[LabeledPoint]](desc:String,
                                                                                                  algorithm:GeneralizedLinearAlgorithm[GLM, Function],
                                                                                                  solver:Optimizer[LabeledPoint, Function],
                                                                                                  reg:RegularizationContext,
                                                                                                  data:Iterator[(Double, Vector[Double])],
                                                                                                  validator:ModelValidator[GLM]) = {
    runGeneralizedLinearAlgorithmScenario(desc, algorithm , solver, reg, List(1.0), None, NormalizationType.NO_SCALING, data, validator)
  }

  /**
   * Enumerate sets of (description, generalized linear algorithm, invalid data set) tuples.
   *
   * <ul>
   *   <li>Everything should detect NaN/+/-Inf labels</li>
   *   <li>Binary regression should handle labels that are finite but not 0 or 1 (use linear regression data set for this)</li>
   *   <li>Poisson regression should handle negative labels (again, believe linear regression can be used for this)</li>
   * </ul>
   */
  private def getGeneralizedLinearAlgorithmsInvalidLabels() : Array[Tuple4[Object, Object, Object, Object]] = {
    return Array(
      Tuple4("Linear regression, NaN/Inf labels",
        new LinearRegressionAlgorithm(),
        drawSampleFromInvalidLabels(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLES, BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLE_DIMENSIONS),
        new PredictionFiniteValidator()),
      Tuple4("Poisson regression, NaN/Inf labels",
        new PoissonRegressionAlgorithm(),
        drawSampleFromInvalidLabels(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLES, BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLE_DIMENSIONS),
        new PredictionFiniteValidator()),
      Tuple4("Logistic regression, NaN/Inf labels",
        new LogisticRegressionAlgorithm(),
        drawSampleFromInvalidLabels(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLES, BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLE_DIMENSIONS),
        new CompositeModelValidator[LogisticRegressionModel](new PredictionFiniteValidator(), new BinaryPredictionValidator[LogisticRegressionModel]())),
      Tuple4("Logistic regression, non-binary labels",
        new LogisticRegressionAlgorithm(),
        drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLES, BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLE_DIMENSIONS),
        new CompositeModelValidator[LogisticRegressionModel](new PredictionFiniteValidator(), new BinaryPredictionValidator[LogisticRegressionModel]())),
      Tuple4("Poisson regression, negative labels",
        new PoissonRegressionAlgorithm(),
        drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(BaseGLMIntegTest.RANDOM_SEED,
          BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLES, BaseGLMIntegTest.NUMBER_OF_INVALID_SAMPLE_DIMENSIONS),
        new CompositeModelValidator[PoissonRegressionModel](new PredictionNonNegativeValidator(), new NonNegativePredictionValidator[PoissonRegressionModel]())))
  }

  @DataProvider
  def generateInvalidLabelCases() : Array[Array[Object]] = {
    val toGenerate = getGeneralizedLinearAlgorithmsInvalidLabels

    toGenerate.map( x => {
      Array(
        s"Invalid features [$x._1]", // Description
        x._2,                        // Algorithm
        new LBFGS(),                 // Solver
        L2RegularizationContext,     // Regularization
        x._3,                        // data
        x._4                         // validator
      )
    }).toArray
  }

  @Test(dataProvider = "generateInvalidLabelCases", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def checkInvalidLabels[GLM <: GeneralizedLinearModel, Function <: DiffFunction[LabeledPoint]](desc:String,
                                                                                                  algorithm:GeneralizedLinearAlgorithm[GLM, Function],
                                                                                                  solver:Optimizer[LabeledPoint, Function],
                                                                                                  reg:RegularizationContext,
                                                                                                  data:Iterator[(Double, Vector[Double])],
                                                                                                  validator:ModelValidator[GLM]) = {
    runGeneralizedLinearAlgorithmScenario(desc, algorithm , solver, reg, List(1.0), None, NormalizationType.NO_SCALING, data, validator)
  }

  /**
   * Expectation is that there may be many clients of this method (e.g. several different tests that change bindings /
   * data providers to exercise different paths) -- hopefully the majority of tests can be constructed by creating
   * the right bindings.
   */
  def runGeneralizedLinearAlgorithmScenario[GLM <: GeneralizedLinearModel, Function <: DiffFunction[LabeledPoint]](desc:String,
                                                                                                                   algorithm:GeneralizedLinearAlgorithm[GLM, Function],
                                                                                                                   solver:Optimizer[LabeledPoint, Function],
                                                                                                                   reg:RegularizationContext,
                                                                                                                   lambdas:List[Double],
                                                                                                                   summary:Option[BasicStatisticalSummary],
                                                                                                                   norm:NormalizationType,
                                                                                                                   data:Iterator[(Double, Vector[Double])],
                                                                                                                   validator:ModelValidator[GLM]) = sparkTest(desc) {
    // Step 0: configure the algorithm
    algorithm.enableIntercept = true
    algorithm.isTrackingState = true
    algorithm.validateData = true
    algorithm.targetStorageLevel = StorageLevel.MEMORY_ONLY

    // Step 1: generate our input RDD
    val trainingSet:RDD[LabeledPoint] = sc.parallelize(data.map( x => { new LabeledPoint(label = x._1, features = x._2)}).toList)

    // Step 2: actually run
    val models:List[GLM] = algorithm.run(trainingSet, solver, reg, lambdas, norm, summary)

    // Step 3: check convergence
    // TODO: Figure out if this test continues to make sense when we have multiple lambdas and, if not, how it should
    // TODO: be fixed.
    assertTrue(None != solver.getStatesTracker, "State tracking was enabled")
    OptimizerIntegTest.checkConvergence(solver.getStatesTracker.get)

    // Step 4: validate the models
    models.foreach( m => {
      m.validateCoefficients
      validator.validateModelPredictions(m, trainingSet)
    })
  }

  /**
   * Check that validators looking for invalid offsets work
   */
  def runInvalidOffsetScenario[GLM <: GeneralizedLinearModel, Function <: DiffFunction[LabeledPoint]](desc:String,
                                                                                                   algorithm:GeneralizedLinearAlgorithm[GLM, Function],
                                                                                                   solver:Optimizer[LabeledPoint, Function],
                                                                                                   reg:RegularizationContext,
                                                                                                   lambdas:List[Double],
                                                                                                   summary:Option[BasicStatisticalSummary],
                                                                                                   norm:NormalizationType,
                                                                                                   data:Iterator[(Double, Vector[Double])],
                                                                                                   validator:ModelValidator[GLM]) = sparkTest(desc) {
    // Step 0: configure the algorithm
    algorithm.enableIntercept = true
    algorithm.isTrackingState = true
    algorithm.validateData = true
    algorithm.targetStorageLevel = StorageLevel.MEMORY_ONLY

    // Step 1: generate our input RDD
    val trainingSet:RDD[LabeledPoint] = sc.parallelize(data.map( x => { new LabeledPoint(label = x._1, features = x._2, offset = Double.NaN)}).toList)

    // Step 2: actually run
    val models:List[GLM] = algorithm.run(trainingSet, solver, reg, lambdas, norm, summary)
  }

}

/**
 * Mostly constants controlling this test
 */
object BaseGLMIntegTest {
  val RANDOM_SEED:Int = 0
  val NUMBER_OF_SAMPLES:Int = 100000
  val NUMBER_OF_DIMENSIONS:Int = 100
  /** For invalid cases, we can use many fewer samples */
  val NUMBER_OF_INVALID_SAMPLES:Int = 100
  /** For invalid cases, we want much higher dimensionality */
  val NUMBER_OF_INVALID_SAMPLE_DIMENSIONS = 10000
  /** Minimum required AUROC */
  val MINIMUM_CLASSIFIER_AUCROC:Double = 0.95
  /** Maximum allowable magnitude difference between predictions and labels for regression problems (this corresponds to 10 sigma, i.e. events that should occur at most once in the lifespan of our solar system)*/
  val MAXIMUM_ERROR_MAGNITUDE:Double = 10 * SparkTestUtils.INLIER_STANDARD_DEVIATION

}