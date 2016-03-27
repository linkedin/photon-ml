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
package com.linkedin.photon.ml

import java.io.File

import breeze.linalg.{DenseVector, Vector, norm}
import com.linkedin.photon.ml.OptionNames._
import com.linkedin.photon.ml.io.{FieldNamesType, GLMSuite}
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.optimization.{OptimizerType, RegularizationType}
import com.linkedin.photon.ml.optimization.RegularizationType.RegularizationType
import com.linkedin.photon.ml.supervised.TaskType
import com.linkedin.photon.ml.supervised.TaskType.TaskType
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils, TestTemplateWithTmpDir}
import org.apache.commons.io.FileUtils
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import scala.collection.mutable
import scala.io.Source

/**
 * This class tests Driver with a set of important configuration parameters
 *
 * @author yizhou
 * @author dpeng
 * @author bdrew
 */
class DriverIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import DriverIntegTest._

  @Test
  def testRunWithMinimalArguments(): Unit = sparkTestSelfServeContext("testRunWithMinimalArguments") {
    val tmpDir = getTmpDir + "/testRunWithMinimalArguments"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir)
    args += CommonTestUtils.fromOptionNameToArg(OPTIMIZER_TYPE_OPTION)
    args += "TRON"

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
  def testRunWithOffHeapMap(): Unit = sparkTestSelfServeContext("testRunWithMinimalArguments") {
    val tmpDir = getTmpDir + "/testRunWithMinimalArguments"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir)
    args += CommonTestUtils.fromOptionNameToArg(OPTIMIZER_TYPE_OPTION)
    args += "TRON"
    args += CommonTestUtils.fromOptionNameToArg(INTERCEPT_OPTION)
    args += "false"
    appendOffHeapConfig(args, addIntercept = false)

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED),
      expectedNumFeatures = 13, expectedNumTrainingData = 250, expectedIsSummarized = false)

    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)
    assertEquals(models.size, 4)
    // Verify lambdas
    assertEquals(models.map(_._1), Array(0.1, 1, 10, 100))

    // No best model output dir
    assertFalse(new File(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT).exists())
  }

  @Test
  def testRunWithOffHeapMapWithIntercept(): Unit = sparkTestSelfServeContext("testRunWithMinimalArguments") {
    val tmpDir = getTmpDir + "/testRunWithMinimalArguments"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir)
    args += CommonTestUtils.fromOptionNameToArg(OPTIMIZER_TYPE_OPTION)
    args += "TRON"
    appendOffHeapConfig(args, addIntercept = true)

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
  def testRunWithDataValidationPerIterationWithOffHeapMap(): Unit = sparkTestSelfServeContext(
    "testRunWithDataValidationPerIteration") {
    val tmpDir = getTmpDir + "/testRunWithDataValidationPerIteration"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir, isValidating = true)

    args += CommonTestUtils.fromOptionNameToArg(VALIDATE_PER_ITERATION)
    args += true.toString()
    args += CommonTestUtils.fromOptionNameToArg(INTERCEPT_OPTION)
    args += "false"
    appendOffHeapConfig(args, addIntercept = false)

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED, DriverStage.VALIDATED, DriverStage.DIAGNOSED),
      expectedNumFeatures = 13, expectedNumTrainingData = 250, expectedIsSummarized = false)

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

  @Test
  def testRunWithDataValidationPerIterationWithOffHeapMapWithIntercept(): Unit = sparkTestSelfServeContext(
    "testRunWithDataValidationPerIteration") {
    val tmpDir = getTmpDir + "/testRunWithDataValidationPerIteration"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir, isValidating = true)

    args += CommonTestUtils.fromOptionNameToArg(VALIDATE_PER_ITERATION)
    args += true.toString()
    appendOffHeapConfig(args, addIntercept = true)

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED, DriverStage.VALIDATED, DriverStage.DIAGNOSED),
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

  @Test
  def testRunWithTRON(): Unit = sparkTestSelfServeContext("testRunWithTRON") {
    val tmpDir = getTmpDir + "/testRunWithTRON"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir)
    args += CommonTestUtils.fromOptionNameToArg(OPTIMIZER_TYPE_OPTION)
    args += OptimizerType.TRON.toString()

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
  def testRunWithLBFGS(): Unit = sparkTestSelfServeContext("testRunWithLBFGS") {
    val tmpDir = getTmpDir + "/testRunWithLBFGS"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir)
    args += CommonTestUtils.fromOptionNameToArg(OPTIMIZER_TYPE_OPTION)
    args += OptimizerType.LBFGS.toString()

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
    args += NormalizationType.SCALE_WITH_STANDARD_DEVIATION.toString()

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = true)

    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)
    assertEquals(models.size, 4)
    // Verify lambdas
    assertEquals(models.map(_._1), Array(0.1, 1, 10, 100))

    // No best model output dir
    assertFalse(new File(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT).exists())
  }

  @Test
  def testRuntWithFeatureStandardization(): Unit = sparkTestSelfServeContext("testRuntWithFeatureScaling") {
    val tmpDir = getTmpDir + "/testRuntWithFeatureNormalization"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir)
    args += CommonTestUtils.fromOptionNameToArg(NORMALIZATION_TYPE)
    args += NormalizationType.STANDARDIZATION.toString()

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = true)
    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)
    assertEquals(models.size, 4)
    // Verify lambdas
    assertEquals(models.map(_._1), Array(0.1, 1, 10, 100))

    // No best model output dir
    assertFalse(new File(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT).exists())
  }

  @Test
  def testRuntWithTreeAggregate(): Unit = sparkTestSelfServeContext("testRuntWithTreeAggregate") {
    val tmpDir = getTmpDir + "/testRuntWithTreeAggregate"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir)

    args += CommonTestUtils.fromOptionNameToArg(TREE_AGGREGATE_DEPTH)
    args += 2.toString

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
    val tmpDir = getTmpDir + "/testRunWithDataValidation"
    val args = mutable.ArrayBuffer[String]()
    appendCommonJobArgs(args, tmpDir, isValidating = true)
    args += CommonTestUtils.fromOptionNameToArg(SUMMARIZATION_OUTPUT_DIR)
    args += tmpDir + "/summary"

    args += CommonTestUtils.fromOptionNameToArg(NORMALIZATION_TYPE)
    args += NormalizationType.SCALE_WITH_STANDARD_DEVIATION.toString()

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED, DriverStage.VALIDATED, DriverStage.DIAGNOSED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = true)

    val models = loadAllModels(tmpDir + "/output/" + Driver.LEARNED_MODELS_TEXT)
    assertEquals(models.size, 4)
    // Verify lambdas
    assertEquals(models.map(_._1), Array(0.1, 1, 10, 100))

    // The selected best model is supposed to be of lambda 100.0 with features scaling with standard deviation
    val bestModel = loadAllModels(tmpDir + "/output/" + Driver.BEST_MODEL_TEXT)
    assertEquals(bestModel.size, 1)
    // Verify lambda
    assertEquals(bestModel(0)._1, 100.0)
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
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED, DriverStage.VALIDATED, DriverStage.DIAGNOSED),
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

  @DataProvider
  def testInvalidRegularizationAndOptimizerDataProvider(): Array[Array[Any]] = {
    Array(
      Array(RegularizationType.L1, OptimizerType.TRON),
      Array(RegularizationType.ELASTIC_NET, OptimizerType.TRON)
    )
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
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED, DriverStage.VALIDATED, DriverStage.DIAGNOSED),
      expectedNumFeatures = 14, expectedNumTrainingData = 250, expectedIsSummarized = false)
  }

  @DataProvider
  def testDiagnosticGenerationProvider(): Array[Array[Any]] = {
    val base = getClass.getClassLoader.getResource("DriverIntegTest/input").getPath
    val models = Map(
      TaskType.LINEAR_REGRESSION ->("linear_regression_train.avro", "linear_regression_val.avro", 7, 1000),
      TaskType.LOGISTIC_REGRESSION ->("logistic_regression_train.avro", "logistic_regression_val.avro", 124, 32561),
      TaskType.POISSON_REGRESSION ->("poisson_train.avro", "poisson_test.avro", 27, 13547)

      // Note: temporarily disabled due to OFFREL-934. Details explained in the ticket.
      //      , TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM ->("logistic_regression_train.avro", "logistic_regression_val.avro", 124, 32561)
    )

    val lambdas = List(0, 1, 10, 100, 1000, 10000)

    val regularizations = Map(
      RegularizationType.NONE ->(OptimizerType.TRON, List(0.0)),
      RegularizationType.L2 ->(OptimizerType.TRON, lambdas),
      RegularizationType.L1 ->(OptimizerType.LBFGS, lambdas),
      RegularizationType.ELASTIC_NET ->(OptimizerType.LBFGS, lambdas))

    (for (m <- models; r <- regularizations) yield {
      (m._1, m._2._1, m._2._2, r._1, r._2._1, r._2._2, m._2._3, m._2._4)
    }).map(x => {
      val (taskType, trainData, testData, regType, optimType, lambdas, numDim, numSamp) = x
      val trainEnabled = (TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM != taskType)

      val outputDir = s"${taskType}_${regType}_${trainEnabled}"

      val args = mutable.ArrayBuffer[String]()
      args += CommonTestUtils.fromOptionNameToArg(TOLERANCE_OPTION)
      if (taskType == TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM) {
        args += 1e-1.toString
      } else {
        args += 1e-6.toString
      }

      args += CommonTestUtils.fromOptionNameToArg(MAX_NUM_ITERATIONS_OPTION)
      if (taskType == TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM) {
        args += 10.toString
      } else {
        args += 200.toString
      }

      args += CommonTestUtils.fromOptionNameToArg(REGULARIZATION_TYPE_OPTION)
      args += regType.toString
      args += CommonTestUtils.fromOptionNameToArg(OPTIMIZER_TYPE_OPTION)
      args += optimType.toString
      args += CommonTestUtils.fromOptionNameToArg(TRAIN_DIR_OPTION)
      args += s"$base/$trainData"
      args += CommonTestUtils.fromOptionNameToArg(OUTPUT_DIR_OPTION)
      args += outputDir + "/models"
      args += CommonTestUtils.fromOptionNameToArg(TASK_TYPE_OPTION)
      args += taskType.toString
      args += CommonTestUtils.fromOptionNameToArg(FORMAT_TYPE_OPTION)
      if (TaskType.POISSON_REGRESSION == taskType) {
        args += FieldNamesType.RESPONSE_PREDICTION.toString
      } else {
        args += FieldNamesType.TRAINING_EXAMPLE.toString
      }
      args += CommonTestUtils.fromOptionNameToArg(VALIDATE_DIR_OPTION)
      args += s"$base/$testData"
      args += CommonTestUtils.fromOptionNameToArg(REGULARIZATION_WEIGHTS_OPTION)
      args += lambdas.mkString(",")
      args += CommonTestUtils.fromOptionNameToArg(ELASTIC_NET_ALPHA_OPTION)
      args += 0.5.toString
      args += CommonTestUtils.fromOptionNameToArg(SUMMARIZATION_OUTPUT_DIR)
      args += outputDir + "/summary"
      args += CommonTestUtils.fromOptionNameToArg(NORMALIZATION_TYPE)
      args += NormalizationType.STANDARDIZATION.toString
      args += CommonTestUtils.fromOptionNameToArg(TRAINING_DIAGNOSTICS)
      args += trainEnabled.toString
      Array(outputDir, args.toArray, numDim, numSamp)
    }).toArray
  }

  @Test(dataProvider = "testDiagnosticGenerationProvider")
  def testDiagnosticGeneration(outputDir: String, args: Array[String], numFeatures: Int, numTrainingSamples: Int): Unit = sparkTestSelfServeContext("testDiagnosticGeneration") {
    FileUtils.deleteDirectory(new File(outputDir))

    MockDriver.runLocally(args = args.toArray,
      expectedStages = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED, DriverStage.VALIDATED, DriverStage.DIAGNOSED),
      expectedNumFeatures = numFeatures, expectedNumTrainingData = numTrainingSamples, expectedIsSummarized = true)
  }
}

object DriverIntegTest {
  val TEST_DIR = ClassLoader.getSystemResource("DriverIntegTest").getPath
  val OFFHEAP_HEART_STORE_NO_INTERCEPT = TEST_DIR + "/paldb_offheapmap_for_heart"
  val OFFHEAP_HEART_STORE_WITH_INTERCEPT = TEST_DIR + "/paldb_offheapmap_for_heart_with_intercept"
  val OFFHEAP_HEART_STORE_PARTITION_NUM = "2"

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

  def appendOffHeapConfig(args: mutable.ArrayBuffer[String], addIntercept: Boolean = true): Unit = {
    args += CommonTestUtils.fromOptionNameToArg(OFFHEAP_INDEXMAP_DIR)
    if (addIntercept) {
      args += OFFHEAP_HEART_STORE_WITH_INTERCEPT
    } else {
      args += OFFHEAP_HEART_STORE_NO_INTERCEPT
    }
    args += CommonTestUtils.fromOptionNameToArg(OFFHEAP_INDEXMAP_NUM_PARTITIONS)
    args += OFFHEAP_HEART_STORE_PARTITION_NUM
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
