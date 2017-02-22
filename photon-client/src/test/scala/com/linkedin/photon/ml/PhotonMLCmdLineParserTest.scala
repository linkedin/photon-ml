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

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.diagnostics.DiagnosticMode
import com.linkedin.photon.ml.io.FieldNamesType
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.{OptimizerType, RegularizationType}
import com.linkedin.photon.ml.PhotonOptionNames._
import com.linkedin.photon.ml.test.CommonTestUtils

/**
 * This class tests PhotonMLCmdLineParser, verifying that all params castings and verifications are good.
 */
class PhotonMLCmdLineParserTest {
  import com.linkedin.photon.ml.PhotonMLCmdLineParserTest._

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testMissingRequiredArgTrainDir(): Unit = {
    PhotonMLCmdLineParser.parseFromCommandLine(requiredArgsMissingOne(TRAIN_DIR_OPTION))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testMissingRequiredArgOutputDir(): Unit = {
    PhotonMLCmdLineParser.parseFromCommandLine(requiredArgsMissingOne(OUTPUT_DIR_OPTION))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testMissingRequiredArgTaskType(): Unit = {
    PhotonMLCmdLineParser.parseFromCommandLine(requiredArgsMissingOne(TASK_TYPE_OPTION))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testDuplicatedArgs(): Unit = {
    val args = requiredArgs()
    val duplicatedArgs = args ++ Array(CommonTestUtils.fromOptionNameToArg(TRAIN_DIR_OPTION), "duplicate")
    PhotonMLCmdLineParser.parseFromCommandLine(duplicatedArgs)
  }

  @Test
  def testPresentingAllRequiredArgs(): Unit = {
    val params = PhotonMLCmdLineParser.parseFromCommandLine(requiredArgs())

    assertEquals(params.jobName, s"Photon-ML-Training")
    // Verify required parameters values
    assertEquals(params.trainDir, "value")
    assertEquals(params.outputDir, "value")
    assertEquals(params.taskType, TaskType.LINEAR_REGRESSION)

    // Verify optional parameters values, should be default values
    assertEquals(params.validateDirOpt, defaultParams.validateDirOpt)
    assertEquals(params.maxNumIter, defaultParams.maxNumIter)
    assertEquals(params.regularizationWeights, defaultParams.regularizationWeights)
    assertEquals(params.tolerance, defaultParams.tolerance)
    assertEquals(params.optimizerType, defaultParams.optimizerType)
    assertEquals(params.regularizationType, defaultParams.regularizationType)
    assertEquals(params.elasticNetAlpha, defaultParams.elasticNetAlpha)
    assertEquals(params.addIntercept, defaultParams.addIntercept)
    assertEquals(params.enableOptimizationStateTracker, defaultParams.enableOptimizationStateTracker)
    assertEquals(params.validatePerIteration, defaultParams.validatePerIteration)
    assertEquals(params.minNumPartitions, defaultParams.minNumPartitions)
    assertEquals(params.kryo, defaultParams.kryo)
    assertEquals(params.fieldsNameType, defaultParams.fieldsNameType)
    assertEquals(params.summarizationOutputDirOpt, defaultParams.summarizationOutputDirOpt)
    assertEquals(params.normalizationType, defaultParams.normalizationType)
    assertEquals(params.constraintString, defaultParams.constraintString)
    assertEquals(params.treeAggregateDepth, defaultParams.treeAggregateDepth)
  }

  @Test
  def testPresentAllArgs(): Unit = {
    val params = PhotonMLCmdLineParser.parseFromCommandLine(requiredArgs() ++ optionalArgs())

    assertEquals(params.jobName, "Job Foo")
    // Verify required parameters values
    assertEquals(params.trainDir, "value")
    assertEquals(params.outputDir, "value")
    assertEquals(params.taskType, TaskType.LINEAR_REGRESSION)

    // Verify optional parameters values
    assertEquals(params.validateDirOpt, Some("validate_dir"))
    assertEquals(params.maxNumIter, 3)
    assertEquals(params.regularizationWeights, List(0.5, 0.7))
    assertEquals(params.tolerance, 1e-3)
    assertEquals(params.optimizerType, OptimizerType.TRON)
    assertEquals(params.regularizationType, RegularizationType.L2)
    assertEquals(params.elasticNetAlpha, Some(0.5))
    assertTrue(params.addIntercept)
    assertTrue(params.enableOptimizationStateTracker)
    assertTrue(params.validatePerIteration)
    assertEquals(params.minNumPartitions, 888)
    assertTrue(params.kryo)
    assertEquals(params.fieldsNameType, FieldNamesType.TRAINING_EXAMPLE)
    assertEquals(params.summarizationOutputDirOpt, Some("summarization_output_dir"))
    assertEquals(params.normalizationType, NormalizationType.NONE)
    assertEquals(params.constraintString, Some(constraintString))
    assertEquals(params.treeAggregateDepth, 2)

    val params2 = PhotonMLCmdLineParser.parseFromCommandLine(requiredArgs() ++ optionalArgs(booleanOptionValue = false))

    assertEquals(params2.jobName, "Job Foo")
    // Verify required parameters values
    assertEquals(params2.trainDir, "value")
    assertEquals(params2.outputDir, "value")
    assertEquals(params2.taskType, TaskType.LINEAR_REGRESSION)

    // Verify optional parameters values
    assertEquals(params2.validateDirOpt, Some("validate_dir"))
    assertEquals(params2.maxNumIter, 3)
    assertEquals(params2.regularizationWeights, List(0.5, 0.7))
    assertEquals(params2.tolerance, 1e-3)
    assertEquals(params2.optimizerType, OptimizerType.TRON)
    assertEquals(params2.regularizationType, RegularizationType.L2)
    assertFalse(params2.addIntercept)
    assertFalse(params2.enableOptimizationStateTracker)
    assertFalse(params2.validatePerIteration)
    assertEquals(params2.minNumPartitions, 888)
    assertFalse(params2.kryo)
    assertEquals(params2.fieldsNameType, FieldNamesType.TRAINING_EXAMPLE)
    assertEquals(params2.summarizationOutputDirOpt, Some("summarization_output_dir"))
    assertEquals(params2.normalizationType, NormalizationType.NONE)
    assertEquals(params2.constraintString, Some(constraintString))
    assertEquals(params2.treeAggregateDepth, 2)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testL1RegularizationAndTRON(): Unit = {
    val rawArgs = requiredArgs()
    val invalidArgs = rawArgs ++
        Array(CommonTestUtils.fromOptionNameToArg(OPTIMIZER_TYPE_OPTION), OptimizerType.TRON.toString,
          CommonTestUtils.fromOptionNameToArg(REGULARIZATION_TYPE_OPTION), RegularizationType.L1.toString)
    PhotonMLCmdLineParser.parseFromCommandLine(invalidArgs)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testElasticNetRegularizationAndTRON(): Unit = {
    val rawArgs = requiredArgs()
    val invalidArgs = rawArgs ++
        Array(CommonTestUtils.fromOptionNameToArg(OPTIMIZER_TYPE_OPTION), OptimizerType.TRON.toString,
          CommonTestUtils.fromOptionNameToArg(REGULARIZATION_TYPE_OPTION), RegularizationType.ELASTIC_NET.toString)
    PhotonMLCmdLineParser.parseFromCommandLine(invalidArgs)
  }

  @Test
  def testNoneRegularizationOverrideDefaultRegularizationWeight(): Unit = {
    val rawArgs = requiredArgs()
    val noneRegularization = rawArgs ++
        Array(CommonTestUtils.fromOptionNameToArg(REGULARIZATION_TYPE_OPTION), RegularizationType.NONE.toString)
    val weights = PhotonMLCmdLineParser.parseFromCommandLine(noneRegularization).regularizationWeights
    assertEquals(weights, List(0.0))
  }

  @Test
  def testNoneRegularizationOverrideRegularizationWeight(): Unit = {
    val rawArgs = requiredArgs()
    val noneRegularization = rawArgs ++
        Array(CommonTestUtils.fromOptionNameToArg(REGULARIZATION_TYPE_OPTION), RegularizationType.NONE.toString,
          CommonTestUtils.fromOptionNameToArg(REGULARIZATION_WEIGHTS_OPTION), Array(0.2, 1.0).mkString(","))
    val weights = PhotonMLCmdLineParser.parseFromCommandLine(noneRegularization).regularizationWeights
    assertEquals(weights, List(0.0))
  }

  @Test
  def testDuplicateRegularizationWeights(): Unit = {
    val rawArgs = requiredArgs()
    val regularization = rawArgs ++
      Array(CommonTestUtils.fromOptionNameToArg(REGULARIZATION_TYPE_OPTION), RegularizationType.L1.toString,
        CommonTestUtils.fromOptionNameToArg(REGULARIZATION_WEIGHTS_OPTION), Array(0.2, 1.0, 1.0, 2.0).mkString(","))
    val weights = PhotonMLCmdLineParser.parseFromCommandLine(regularization).regularizationWeights
    assertEquals(weights, List(0.2, 1.0, 2.0))
  }

  @DataProvider
  def generateUnparseableConstraintStrings(): Array[Array[Object]] = {
    Array(
      // unparseable
      Array("""{"name": "ageInHour", "term": "", "upperBound": 0 """),
      // parseable but not an array of maps
      Array("""{"name": "ageInHour", "term": "", "upperBound": 0, "lowerBound": -1}""")
    )
  }

  @Test(dataProvider = "generateUnparseableConstraintStrings",
    expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testUnparseableConstrainedString(constraintString: String): Unit = {
    val args = new Array[String](2)
    args(0) = CommonTestUtils.fromOptionNameToArg(COEFFICIENT_BOX_CONSTRAINTS)
    args(1) = constraintString
    PhotonMLCmdLineParser.parseFromCommandLine(requiredArgs() ++ args)
  }

  @Test
  def testParseableConstrainedString(): Unit = {
    val args = new Array[String](2)
    args(0) = CommonTestUtils.fromOptionNameToArg(COEFFICIENT_BOX_CONSTRAINTS.toString)
    args(1) = PhotonMLCmdLineParserTest.constraintString
    assertEquals(PhotonMLCmdLineParser.parseFromCommandLine(requiredArgs() ++ args).constraintString,
                 Some(PhotonMLCmdLineParserTest.constraintString))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testFeatureStandardizationWithNoIntercept(): Unit = {
    val args = Array(
      CommonTestUtils.fromOptionNameToArg(NORMALIZATION_TYPE),
      NormalizationType.STANDARDIZATION.toString,
      CommonTestUtils.fromOptionNameToArg(INTERCEPT_OPTION),
      false.toString
    )
    PhotonMLCmdLineParser.parseFromCommandLine(requiredArgs() ++ args)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidTreeAggregateDepth(): Unit = {
    val args = Array(
      CommonTestUtils.fromOptionNameToArg(TREE_AGGREGATE_DEPTH),
      0.toString
    )
    PhotonMLCmdLineParser.parseFromCommandLine(requiredArgs() ++ args)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testDiagnosticModeBeingALLWhenNoValidateDirSpecified(): Unit = {
    val args = Array(
      CommonTestUtils.fromOptionNameToArg(DIAGNOSTIC_MODE),
      DiagnosticMode.ALL.toString
    )
    PhotonMLCmdLineParser.parseFromCommandLine(requiredArgs() ++ args)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testDiagnosticModeBeingValidateWhenNoValidateDirSpecified(): Unit = {
    val args = Array(
      CommonTestUtils.fromOptionNameToArg(DIAGNOSTIC_MODE),
      DiagnosticMode.VALIDATE.toString
    )
    PhotonMLCmdLineParser.parseFromCommandLine(requiredArgs() ++ args)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testDiagnosticModeAndTrainDiagnosticsBothSpecified(): Unit = {
    val args = Array(
      CommonTestUtils.fromOptionNameToArg(DIAGNOSTIC_MODE),
      DiagnosticMode.NONE.toString,
      CommonTestUtils.fromOptionNameToArg(TRAINING_DIAGNOSTICS),
      "false"
    )
    PhotonMLCmdLineParser.parseFromCommandLine(requiredArgs() ++ args)
  }
}

object PhotonMLCmdLineParserTest {

  private val defaultParams = new Params
  private val REQUIRED_OPTIONS = Array(TRAIN_DIR_OPTION, OUTPUT_DIR_OPTION, TASK_TYPE_OPTION)

  // Optional options other than boolean options
  private val OPTIONAL_OPTIONS = Array(VALIDATE_DIR_OPTION,
    REGULARIZATION_WEIGHTS_OPTION,
    REGULARIZATION_TYPE_OPTION,
    ELASTIC_NET_ALPHA_OPTION,
    MAX_NUM_ITERATIONS_OPTION,
    TOLERANCE_OPTION,
    JOB_NAME_OPTION,
    OPTIMIZER_TYPE_OPTION,
    FORMAT_TYPE_OPTION,
    MIN_NUM_PARTITIONS_OPTION,
    SUMMARIZATION_OUTPUT_DIR,
    NORMALIZATION_TYPE,
    COEFFICIENT_BOX_CONSTRAINTS,
    TREE_AGGREGATE_DEPTH,
    DIAGNOSTIC_MODE
  )

  // Boolean options are unary instead of binary options
  private val BOOLEAN_OPTIONAL_OPTIONS = Array(
    INTERCEPT_OPTION,
    KRYO_OPTION,
    VALIDATE_PER_ITERATION,
    OPTIMIZATION_STATE_TRACKER_OPTION
  )

  private val constraintString =
    """[
             {"name": "ageInHour", "term": "", "lowerBound": -1, "upperBound": 0},
             {"name": "ageInHour:lv", "term": "4", "lowerBound": -1},
             {"name": "ageInHour:lv", "term": "12", "upperBound": 0},
             {"name": "actor_rawclicksw.gt.0", "term": "*", "lowerBound": -0.01}
       ]"""

  /**
   *
   * @return
   */
  def requiredArgs(): Array[String] = {
    val args = new Array[String](REQUIRED_OPTIONS.length * 2)
    var i = 0
    REQUIRED_OPTIONS.foreach { option =>
      args(i) = CommonTestUtils.fromOptionNameToArg(option)
      args(i + 1) = option match {
        case TASK_TYPE_OPTION => TaskType.LINEAR_REGRESSION.toString.toLowerCase
        case _ => "value"
      }
      i += 2
    }
    args
  }

  /**
   *
   * @param booleanOptionValue
   * @return
   */
  def optionalArgs(booleanOptionValue: Boolean = true): Array[String] = {
    val args = new Array[String](OPTIONAL_OPTIONS.length * 2 + BOOLEAN_OPTIONAL_OPTIONS.length * 2)
    var i = 0
    OPTIONAL_OPTIONS.foreach { option =>
      args(i) = CommonTestUtils.fromOptionNameToArg(option)
      args(i + 1) = option match {
        case VALIDATE_DIR_OPTION => "validate_dir"
        case REGULARIZATION_WEIGHTS_OPTION => List(0.5, 0.7).mkString(",")
        case REGULARIZATION_TYPE_OPTION => RegularizationType.L2.toString
        case ELASTIC_NET_ALPHA_OPTION => "0.5"
        case MAX_NUM_ITERATIONS_OPTION => 3.toString
        case TOLERANCE_OPTION => 1e-3.toString
        case JOB_NAME_OPTION => "Job Foo"
        case OPTIMIZER_TYPE_OPTION => OptimizerType.TRON.toString
        case FORMAT_TYPE_OPTION => FieldNamesType.TRAINING_EXAMPLE.toString
        case MIN_NUM_PARTITIONS_OPTION => 888.toString
        case SUMMARIZATION_OUTPUT_DIR => "summarization_output_dir"
        case NORMALIZATION_TYPE => NormalizationType.NONE.toString
        case COEFFICIENT_BOX_CONSTRAINTS => constraintString
        case TREE_AGGREGATE_DEPTH => 2.toString
        case DIAGNOSTIC_MODE => DiagnosticMode.NONE.toString
        case _ => "dummy-value"
      }
      i += 2
    }

    BOOLEAN_OPTIONAL_OPTIONS.foreach { option =>
      args(i) = CommonTestUtils.fromOptionNameToArg(option)
      args(i + 1) = booleanOptionValue.toString
      i += 2
    }

    args
  }

  /**
   *
   * @param missingArgName
   * @return
   */
  def requiredArgsMissingOne(missingArgName: String): Array[String] = {
    if (REQUIRED_OPTIONS.isEmpty) {
      throw new RuntimeException("No required option configured in test.")
    }

    val args = new Array[String]((REQUIRED_OPTIONS.length - 1) * 2)
    var i = 0
    REQUIRED_OPTIONS.filter(_ != missingArgName).foreach { option =>
      args(i) = CommonTestUtils.fromOptionNameToArg(option)
      args(i + 1) = option match {
        case TASK_TYPE_OPTION => TaskType.LINEAR_REGRESSION.toString.toLowerCase
        case _ => "value"
      }
      i += 2
    }
    args
  }
}
