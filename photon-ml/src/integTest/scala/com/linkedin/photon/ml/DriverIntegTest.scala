package com.linkedin.photon.ml

import java.util.UUID

import OptionNames._
import com.linkedin.photon.ml.io.FieldNamesType
import com.linkedin.photon.ml.optimization.{OptimizerType, RegularizationType}
import OptimizerType.OptimizerType
import com.linkedin.photon.ml.optimization.RegularizationType
import RegularizationType.RegularizationType
import com.linkedin.photon.ml.supervised.TaskType
import TaskType.TaskType
import com.linkedin.photon.ml.test.SparkTestUtils
import breeze.linalg.Vector
import java.io.File

import com.linkedin.photon.ml.io.{GLMSuite, FieldNamesType}
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.RegularizationType
import com.linkedin.photon.ml.supervised.TaskType
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.{SparkTestUtils, CommonTestUtils, TestTemplateWithTmpDir}

import scala.collection.mutable
import scala.io.Source

import breeze.linalg.{norm, DenseVector}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

/**
 * This class tests Driver with a set of important configuration parameters
 *
 * @author yizhou
 * @author dpeng
 */
class DriverIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import DriverIntegTest._

  @Test
  def testRunWithMinimalArguments(): Unit = sparkTestSelfServeContext("testRunWithMinimalArguments") {
    val tmpDir = getTmpDir + "/testRunWithMinimalArguments"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir)

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = false)

    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)
    assertEquals(models.size, 4)
    // Verify lambdas
    assertEquals(models.map(_._1), Array(0.1, 1, 10, 100))

    // No best model output dir
    assertFalse(new File(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT).exists())
  }

  /**
   * Test training with L1 regularization. This test verify that as the regularization weight increases,
   * coefficients shrink and will be shrink to zero.
   */
  @Test
  def testRunWithL1(): Unit = sparkTestSelfServeContext("testRunWithL1") {
    val tmpDir = getTmpDir + "/testRunWithL1"
    val args = mutable.ArrayBuffer[String]()
    val lambdas = Array(1.0, 10.0, 100.0, 1000.0)
    appendCommonJobArgs(args, tmpDir)
    args += CommonTestUtils.fromOptionNameToArg(REGULARIZATION_TYPE_OPTION)
    args += RegularizationType.L1.toString
    args += CommonTestUtils.fromOptionNameToArg(REGULARIZATION_WEIGHTS_OPTION)
    args += lambdas.mkString(",")
    args += CommonTestUtils.fromOptionNameToArg(INTERCEPT_OPTION)
    args += "false"
    args += CommonTestUtils.fromOptionNameToArg(MAX_NUM_ITERATIONS_OPTION)
    args += "500"
    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED),
      expectedNumFeatures = 13, expectedNumTrainingData = 250, expectedIsSummarized = false)

    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)

    assertEquals(models.size, lambdas.length)
    // Verify lambdas
    assertEquals(models.map(_._1), lambdas)

    // No best model output dir
    assertFalse(new File(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT).exists())

    val epsilon = 1.0E-10
    val normsAndCounts = models.map { case (lambda, model) => (lambda, norm(model.coefficients, 1), model.coefficients.toArray.count(x => Math.abs(x) < epsilon)) }
    for (i <- 0 until lambdas.length - 1) {
      val (lambda1, norm1, zero1) = normsAndCounts(i)
      val (lambda2, norm2, zero2) = normsAndCounts(i + 1)
      assertTrue(norm1 > norm2, s"Norm assertion failed. ($lambda1, $norm1, $zero1) and ($lambda2, $norm2, $zero2)")
      assertTrue(zero1 < zero2, s"Zero count assertion failed. ($lambda1, $norm1, $zero1) and ($lambda2, $norm2, $zero2)")
    }
  }

  /**
   * Test training with elastic net regularization with alpha = 0.5. This test verify that as the regularization weight increases,
   * coefficients shrink and will be shrink to zero.
   */
  @Test
  def testRunWithElasticNet(): Unit = sparkTestSelfServeContext("testRunWithElasticNet") {
    val tmpDir = getTmpDir + "/testRunWithElasticNet"
    val args = mutable.ArrayBuffer[String]()
    val lambdas = Array(10.0, 100.0, 1000.0, 10000.0)
    val alpha = 0.5
    appendCommonJobArgs(args, tmpDir)
    args += CommonTestUtils.fromOptionNameToArg(REGULARIZATION_TYPE_OPTION)
    args += RegularizationType.ELASTIC_NET.toString
    args += CommonTestUtils.fromOptionNameToArg(ELASTIC_NET_ALPHA_OPTION)
    args += alpha.toString
    args += CommonTestUtils.fromOptionNameToArg(REGULARIZATION_WEIGHTS_OPTION)
    args += lambdas.mkString(",")
    args += CommonTestUtils.fromOptionNameToArg(MAX_NUM_ITERATIONS_OPTION)
    args += "500"

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = false)

    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)
    assertEquals(models.size, lambdas.length)
    // Verify lambdas
    assertEquals(models.map(_._1), lambdas)

    // No best model output dir
    assertFalse(new File(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT).exists())

    val epsilon = 1.0E-10
    def elasticNetNorm(vec: Vector[Double], alpha: Double): Double = {
      alpha * norm(vec, 1) + (1 - alpha) * norm(vec, 2)
    }
    val normsAndCounts = models.map {
      case (lambda, model) => (lambda, alpha * elasticNetNorm(model.coefficients, alpha), model.coefficients.toArray.count(x => Math.abs(x) < epsilon))
    }
    for (i <- 0 until lambdas.length - 1) {
      val (lambda1, norm1, zero1) = normsAndCounts(i)
      val (lambda2, norm2, zero2) = normsAndCounts(i + 1)
      assertTrue(norm1 > norm2, s"Norm assertion failed. ($lambda1, $norm1, $zero1) and ($lambda2, $norm2, $zero2)")
      assertTrue(zero1 < zero2, s"Zero count assertion failed. ($lambda1, $norm1, $zero1) and ($lambda2, $norm2, $zero2)")
    }
  }

  @Test
  def testRuntWithFeatureScaling(): Unit = sparkTestSelfServeContext("testRuntWithFeatureScaling") {
    val tmpDir = getTmpDir + "/testRuntWithFeatureScaling"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir)

    args += CommonTestUtils.fromOptionNameToArg(NORMALIZATION_TYPE)
    args += NormalizationType.USE_STANDARD_DEVIATION.toString()

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = false)

    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)
    assertEquals(models.size, 4)
    // Verify lambdas
    assertEquals(models.map(_._1), Array(0.1, 1, 10, 100))

    // No best model output dir
    assertFalse(new File(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT).exists())
  }

  @Test
  def testRunWithSummarization(): Unit = sparkTestSelfServeContext("testRunWithSummarization") {
    val tmpDir = getTmpDir + "/testRunWithSummarization"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir)

    args += CommonTestUtils.fromOptionNameToArg(SUMMARIZATION_OUTPUT_DIR)
    args += tmpDir + "/summary"

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = true)

    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)
    assertEquals(models.size, 4)
    // Verify lambdas
    assertEquals(models.map(_._1), Array(0.1, 1, 10, 100))

    // No best model output dir
    assertFalse(new File(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT).exists())

    // Verify summary output
    assertTrue(new File(tmpDir + "/summary/part-00000.avro").exists())
  }

  @Test
  def testRunWithDataValidation(): Unit = sparkTestSelfServeContext("testRunWithDataValidation") {
    val tmpDir = s"/tmp/${UUID.randomUUID.toString}/testRunWithDataValidation"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir, isValidating = true)
    args += CommonTestUtils.fromOptionNameToArg(SUMMARIZATION_OUTPUT_DIR)
    args += tmpDir + "/summary"

    args += CommonTestUtils.fromOptionNameToArg(NORMALIZATION_TYPE)
    args += NormalizationType.USE_STANDARD_DEVIATION.toString()

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED, DriverStage.VALIDATED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = true)

    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)
    assertEquals(models.size, 4)
    // Verify lambdas
    assertEquals(models.map(_._1), Array(0.1, 1, 10, 100))

    // The selected best model is supposed to be of lambda 0.1
    val bestModel = loadAllModels(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT)
    assertEquals(bestModel.size, 1)
    // Verify lambda
    assertEquals(bestModel(0)._1, 1.0)
  }

  @Test
  def testRunWithDataValidationPerIteration(): Unit = sparkTestSelfServeContext(
    "testRunWithDataValidationPerIteration") {
    val tmpDir = getTmpDir + "/testRunWithDataValidationPerIteration"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir, isValidating = true)

    args += CommonTestUtils.fromOptionNameToArg(VALIDATE_PER_ITERATION)
    args += true.toString()

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED, DriverStage.VALIDATED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = false)

    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)
    assertEquals(models.size, 4)
    // Verify lambdas
    assertEquals(models.map(_._1), Array(0.1, 1, 10, 100))

    // The selected best model is supposed to be of lambda 0.1
    val bestModel = loadAllModels(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT)
    assertEquals(bestModel.size, 1)
    // Verify lambda
    assertEquals(bestModel(0)._1, 0.1)
  }

  @Test(dataProvider = "testInvalidRegularizationAndOptimizerDataProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidRegularizationAndOptimizer(regularizationType: RegularizationType, optimizer: OptimizerType): Unit = sparkTestSelfServeContext("testInvalidRegularizationAndOptimizer") {
    val tmpDir = getTmpDir + "/testInvalidRegularizationAndOptimizer"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir, isValidating = true)
    args += CommonTestUtils.fromOptionNameToArg(REGULARIZATION_TYPE_OPTION)
    args += regularizationType.toString
    args += CommonTestUtils.fromOptionNameToArg(OPTIMIZER_TYPE_OPTION)
    args += optimizer.toString
    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED, DriverStage.VALIDATED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = false)
  }

  @DataProvider(parallel = false)
  def testInvalidRegularizationAndOptimizerDataProvider(): Array[Array[Any]] = {
    Array(
      Array(RegularizationType.L1, OptimizerType.TRON),
      Array(RegularizationType.ELASTIC_NET, OptimizerType.TRON)
    )
  }

  @Test(dataProvider = "testInvalidRegularizationAndAlphaDataProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidRegularizationAndAlpha(regularizationType: RegularizationType, alpha: Double): Unit = sparkTestSelfServeContext("testInvalidRegularizationAndAlpha") {
    val tmpDir = getTmpDir + "/testInvalidRegularizationAndAlpha"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir, isValidating = true)
    args += CommonTestUtils.fromOptionNameToArg(REGULARIZATION_TYPE_OPTION)
    args += regularizationType.toString
    args += CommonTestUtils.fromOptionNameToArg(ELASTIC_NET_ALPHA_OPTION)
    args += alpha.toString
    MockDriver.runLocally(
      args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED, DriverStage.VALIDATED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = false
    )
  }

  @DataProvider(parallel = false)
  def testInvalidRegularizationAndAlphaDataProvider(): Array[Array[Any]] = {
    Array(
      Array(RegularizationType.L1, 0.5),
      Array(RegularizationType.L2, 0.5),
      Array(RegularizationType.ELASTIC_NET, 1.1),
      Array(RegularizationType.ELASTIC_NET, -0.5)
    )
  }
}

object DriverIntegTest {
  val TEST_DIR = ClassLoader.getSystemResource("DriverIntegTest").getPath

  def appendCommonJobArgs(args: mutable.ArrayBuffer[String], testRoot: String, isValidating: Boolean = false): Unit = {
    args += CommonTestUtils.fromOptionNameToArg(TRAIN_DIR_OPTION)
    args += TEST_DIR + "/input/heart.avro"

    if (isValidating) {
      args += CommonTestUtils.fromOptionNameToArg(VALIDATE_DIR_OPTION)
      args += TEST_DIR + "/input/heart_validation.avro"
    }

    args += CommonTestUtils.fromOptionNameToArg(OUTPUT_DIR_OPTION)
    args += testRoot + "/output"

    args += CommonTestUtils.fromOptionNameToArg(TASK_TYPE_OPTION)
    args += TaskType.LOGISTIC_REGRESSION.toString()

    args += CommonTestUtils.fromOptionNameToArg(FORMAT_TYPE_OPTION)
    args += FieldNamesType.TRAINING_EXAMPLE.toString()
  }

  /* TODO: formalize these model loaders in another RB
   * These model loading uitls are temporarily put here. A thorough solution should be provided while refactoring
   * utils of GLMSuite and we should provide general and flexible ways of ser/der model objects
   */
  def loadAllModels(modelsDir: String): Array[(Double, GeneralizedLinearModel)] = {
    val models = mutable.ArrayBuffer[(Double, GeneralizedLinearModel)]()
    val files = new File(modelsDir).listFiles()
    for (file <- files.filter { f => !f.getName().startsWith("_") && !f.getName().startsWith(".") }) {
      models += loadModelFromText(file.getPath, TaskType.LOGISTIC_REGRESSION)
    }
    models.sortWith(_._1 < _._1).toArray
  }

  def loadModelFromText(modelPath: String, taskType: TaskType): (Double, GeneralizedLinearModel) = {
    val coeffs = mutable.ArrayBuffer[(Long, Double)]()
    var intercept: Option[Double] = None
    var lambda: Option[Double] = None
    for (line <- Source.fromFile(modelPath).getLines()) {
      val tokens = line.split("\t")

      // Heart scale dataset's feature names are indices, and they don't have terms.
      // Thus, we are ignoring tokens(1)
      val name = tokens(0)
      if (name == GLMSuite.INTERCEPT_NAME) {
        intercept = Some(tokens(2).toDouble)
      } else {
        val co = (tokens(0).toLong, tokens(2).toDouble)
        coeffs += co
      }
      lambda = Some(tokens(3).toDouble)
    }
    val features = new DenseVector[Double](coeffs.sortWith(_._1 < _._1).map(_._2).toArray)

    val model = taskType match {
      case TaskType.LOGISTIC_REGRESSION => new LogisticRegressionModel(features, intercept)
      case _ => new RuntimeException("Other task type not supported.")
    }
    (lambda.get, model.asInstanceOf[GeneralizedLinearModel])
  }
}
