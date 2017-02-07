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
package com.linkedin.photon.ml.estimators

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.data.{FixedEffectDataConfiguration, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.io.ModelOutputMode
import com.linkedin.photon.ml.optimization.GLMOptimizationConfiguration
import com.linkedin.photon.ml.optimization.game.MFOptimizationConfiguration
import com.linkedin.photon.ml.test.CommonTestUtils._

/**
 * Simple test for GAME training's [[Params]]
 */
class GameParamsTest {

  import GameParamsTest._

  @DataProvider
  def requiredOptions(): Array[Array[Any]] = {
    REQUIRED_OPTIONS.map(optionName => Array[Any](optionName))
  }

  @Test(dataProvider = "requiredOptions", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testMissingRequiredArg(optionName: String): Unit = {
    GameParams.parseFromCommandLine(mapToArray(requiredArgsMissingOne(optionName)))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testDuplicatedArgs(): Unit = {
    val args = requiredArgs()
    val duplicatedArgs = mapToArray(args) ++ Array(fromOptionNameToArg(TRAIN_INPUT_DIRS), "duplicate")
    GameParams.parseFromCommandLine(duplicatedArgs)
  }

  @Test
  def testPresentingAllRequiredArgs(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(requiredArgs()))

    // Verify required parameters values
    assertEquals(params.trainDirs.deep, Array(TRAIN_INPUT_DIRS).deep)
    assertEquals(params.outputDir, OUTPUT_DIR)
    assertEquals(params.featureNameAndTermSetInputPath, FEATURE_NAME_AND_TERM_SET_PATH)
    assertEquals(params.taskType, TaskType.LINEAR_REGRESSION)
    assertEquals(params.updatingSequence, Seq(UPDATING_SEQUENCE))

    // Verify optional parameters values, should be default values
    assertEquals(params.trainDateRangeOpt, defaultParams.trainDateRangeOpt)
    assertEquals(params.trainDateRangeDaysAgoOpt, defaultParams.trainDateRangeDaysAgoOpt)
    assertEquals(params.validateDirsOpt, defaultParams.validateDirsOpt)
    assertEquals(params.validateDateRangeOpt, defaultParams.validateDateRangeOpt)
    assertEquals(params.validateDateRangeDaysAgoOpt, defaultParams.validateDateRangeDaysAgoOpt)
    assertEquals(params.minPartitionsForValidation, defaultParams.minPartitionsForValidation)
    assertEquals(params.featureShardIdToFeatureSectionKeysMap, defaultParams.featureShardIdToFeatureSectionKeysMap)
    assertEquals(params.featureShardIdToInterceptMap, defaultParams.featureShardIdToInterceptMap)
    assertEquals(params.numIterations, defaultParams.numIterations)
    assertEquals(params.fixedEffectOptimizationConfigurations, defaultParams.fixedEffectOptimizationConfigurations)
    assertEquals(params.fixedEffectDataConfigurations, defaultParams.fixedEffectDataConfigurations)
    assertEquals(params.randomEffectOptimizationConfigurations, defaultParams.randomEffectOptimizationConfigurations)
    assertEquals(params.factoredRandomEffectOptimizationConfigurations,
      defaultParams.factoredRandomEffectOptimizationConfigurations)
    assertEquals(params.randomEffectDataConfigurations, defaultParams.randomEffectDataConfigurations)
    assertEquals(params.modelOutputMode, defaultParams.modelOutputMode)
    assertEquals(params.numberOfOutputFilesForRandomEffectModel, defaultParams.numberOfOutputFilesForRandomEffectModel)
    assertEquals(params.deleteOutputDirIfExists, defaultParams.deleteOutputDirIfExists)
    assertEquals(params.applicationName, defaultParams.applicationName)
  }

  @Test
  def testTrainDateRange(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(TRAIN_DATE_RANGE, "20160501-20160531")))
    assertTrue(params.trainDateRangeOpt.isDefined)
    assertEquals(params.trainDateRangeOpt.get, "20160501-20160531")
  }

  @Test
  def testTrainDateRangeDaysAgo(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(TRAIN_DATE_RANGE_DAYS_AGO, "90-6")))
    assertTrue(params.trainDateRangeDaysAgoOpt.isDefined)
    assertEquals(params.trainDateRangeDaysAgoOpt.get, "90-6")
  }

  @Test
  def testValidateInputDirs(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(VALIDATE_INPUT_DIRS, "dir1,dir2")))
    assertTrue(params.validateDirsOpt.isDefined)
    assertEquals(params.validateDirsOpt.get.deep, Array("dir1", "dir2").deep)
  }

  @Test
  def testValidateDateRange(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(VALIDATE_DATE_RANGE, "20160601-20160608")))
    assertTrue(params.validateDateRangeOpt.isDefined)
    assertEquals(params.validateDateRangeOpt.get, "20160601-20160608")
  }

  @Test
  def testValidateDateRangeDaysAgo(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(VALIDATE_DATE_RANGE_DAYS_AGO, "5-1")))
    assertTrue(params.validateDateRangeDaysAgoOpt.isDefined)
    assertEquals(params.validateDateRangeDaysAgoOpt.get, "5-1")
  }

  @Test
  def testMinPartitionsForValidation(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(MIN_PARTITIONS_FOR_VALIDATION, "5")))
    assertEquals(params.minPartitionsForValidation, 5)
  }

  @Test
  def testFeatureShardIdToFeatureSectionKeysMap(): Unit = {
    val argValueInStr = "shardId1:sectionKey1,sectionKey2|shardId2:sectionKey3"
    val expectedValue = Map("shardId1" -> Set("sectionKey1", "sectionKey2"), "shardId2" -> Set("sectionKey3"))
    val params =
      GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(FEATURE_SHARD_ID_TO_FEATURE_SECTION_KEYS_MAP,
        argValueInStr)))
    assertEquals(params.featureShardIdToFeatureSectionKeysMap, expectedValue)
  }

  @Test
  def testFeatureShardIdToInterceptMap(): Unit = {
    val argValueInStr = "shardId1:TrUe|shardId2:fAlSe"
    val expectedValue = Map("shardId1" -> true, "shardId2" -> false)
    val params =
      GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(FEATURE_SHARD_ID_TO_INTERCEPT_MAP, argValueInStr)))
    assertEquals(params.featureShardIdToInterceptMap, expectedValue)
  }

  @Test
  def testNumIterations(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(NUM_ITERATIONS, "5")))
    assertEquals(params.numIterations, 5)
  }

  @Test
  def testUpdatingSequence(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(UPDATING_SEQUENCE, "5,1,4,2,3")))

    assertEquals(params.updatingSequence, Seq("5", "1", "4", "2", "3"))
  }

  @Test
  def testFixedEffectOptimizationConfigurations(): Unit = {
    import GLMOptimizationConfiguration.{SPLITTER => S}

    val config1InStr = s"1${S}2e-2${S}4${S}0.3${S}LBFGS${S}l1"
    val config2InStr = s"5${S}6E-6${S}7${S}0.2${S}TRON${S}L2"
    val argValueInStr = s"fixed1:$config1InStr|fixed2:$config2InStr;fixed1:$config2InStr|fixed2:$config1InStr"
    val params =
      GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(FIXED_EFFECT_OPTIMIZATION_CONFIGURATIONS, argValueInStr)))
    val config1 = GLMOptimizationConfiguration.parseAndBuildFromString(config1InStr)
    val config2 = GLMOptimizationConfiguration.parseAndBuildFromString(config2InStr)
    val expectedValue = Array(
      Map("fixed1" -> config1, "fixed2" -> config2),
      Map("fixed1" -> config2, "fixed2" -> config1))
    assertEquals(params.fixedEffectOptimizationConfigurations.deep, expectedValue.deep)
  }

  @Test
  def testFixedEffectDataConfigurations(): Unit = {
    import FixedEffectDataConfiguration.{SPLITTER => S}

    val config1InStr = s"shardId${S}1"
    val config2InStr = s"shardId${S}1"
    val argValueInStr = s"fixed1:$config1InStr|fixed2:$config2InStr"
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(FIXED_EFFECT_DATA_CONFIGURATIONS, argValueInStr)))
    val config1 = FixedEffectDataConfiguration.parseAndBuildFromString(config1InStr)
    val config2 = FixedEffectDataConfiguration.parseAndBuildFromString(config2InStr)
    val expectedValue = Map("fixed1" -> config1, "fixed2" -> config2)
    assertEquals(params.fixedEffectDataConfigurations, expectedValue)
  }

  @Test
  def testRandomEffectOptimizationConfigurations(): Unit = {
    import GLMOptimizationConfiguration.{SPLITTER => S}

    val config1InStr = s"1${S}2e-2${S}4${S}0.3${S}LBFGS${S}l1"
    val config2InStr = s"5${S}6E-6${S}7${S}0.2${S}TRON${S}L2"
    val argValueInStr = s"random1:$config1InStr|random2:$config2InStr;random1:$config2InStr|random2:$config1InStr"
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(RANDOM_EFFECT_OPTIMIZATION_CONFIGURATIONS,
      argValueInStr)))
    val config1 = GLMOptimizationConfiguration.parseAndBuildFromString(config1InStr)
    val config2 = GLMOptimizationConfiguration.parseAndBuildFromString(config2InStr)
    val expectedValue = Array(
      Map("random1" -> config1, "random2" -> config2),
      Map("random1" -> config2, "random2" -> config1))
    assertEquals(params.randomEffectOptimizationConfigurations.deep, expectedValue.deep)
  }

  @Test
  def testFactoredRandomEffectOptimizationConfigurations(): Unit = {
    import GLMOptimizationConfiguration.{SPLITTER => GS}
    import MFOptimizationConfiguration.{SPLITTER => MS}

    val randomEffectOptConfig1InStr = s"1${GS}2e-2${GS}4${GS}0.3${GS}LBFGS${GS}l1"
    val latentFactorOptConfig1InStr = s"10${GS}1e-2${GS}5${GS}0.1${GS}TRON${GS}l2"
    val mfOptimizationOptConfig1InStr = s"10${MS}20"
    val randomEffectOptConfig2InStr = s"1${GS}2e-2${GS}4${GS}0.3${GS}TRON${GS}l1"
    val latentFactorOptConfig2InStr = s"10${GS}1e-2${GS}5${GS}0.1${GS}LBFGS${GS}l2"
    val mfOptimizationOptConfig2InStr = s"10${MS}20"
    val argValueInStr =
      s"factor1:$randomEffectOptConfig1InStr:$latentFactorOptConfig1InStr:$mfOptimizationOptConfig1InStr|" +
        s"factor2:$randomEffectOptConfig2InStr:$latentFactorOptConfig2InStr:$mfOptimizationOptConfig2InStr;" +
        s"factor1:$randomEffectOptConfig2InStr:$latentFactorOptConfig2InStr:$mfOptimizationOptConfig2InStr|" +
        s"factor2:$randomEffectOptConfig1InStr:$latentFactorOptConfig1InStr:$mfOptimizationOptConfig1InStr"
    val params =
      GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(FACTORED_RANDOM_EFFECT_OPTIMIZATION_CONFIGURATIONS,
        argValueInStr)))
    val randomEffectOptConfig1 = GLMOptimizationConfiguration.parseAndBuildFromString(randomEffectOptConfig1InStr)
    val latentFactorOptConfig1 = GLMOptimizationConfiguration.parseAndBuildFromString(latentFactorOptConfig1InStr)
    val mfOptimizationOptConfig1 = MFOptimizationConfiguration.parseAndBuildFromString(mfOptimizationOptConfig1InStr)
    val randomEffectOptConfig2 = GLMOptimizationConfiguration.parseAndBuildFromString(randomEffectOptConfig2InStr)
    val latentFactorOptConfig2 = GLMOptimizationConfiguration.parseAndBuildFromString(latentFactorOptConfig2InStr)
    val mfOptimizationOptConfig2 = MFOptimizationConfiguration.parseAndBuildFromString(mfOptimizationOptConfig2InStr)
    val expectedValue = Array(
      Map("factor1" -> (randomEffectOptConfig1, latentFactorOptConfig1, mfOptimizationOptConfig1),
        "factor2" -> (randomEffectOptConfig2, latentFactorOptConfig2, mfOptimizationOptConfig2)),
      Map("factor1" -> (randomEffectOptConfig2, latentFactorOptConfig2, mfOptimizationOptConfig2),
        "factor2" -> (randomEffectOptConfig1, latentFactorOptConfig1, mfOptimizationOptConfig1))
    )
    assertEquals(params.factoredRandomEffectOptimizationConfigurations.deep, expectedValue.deep)
  }

  @Test
  def testRandomEffectDataConfigurations(): Unit = {
    import RandomEffectDataConfiguration.{FIRST_LEVEL_SPLITTER => F, SECOND_LEVEL_SPLITTER => S}

    val config1InStr = s"randomEffectType${F}featureShardId${F}1${F}10${F}5${F}20d${F}random${S}5"
    val config2InStr = s"randomEffectType${F}featureShardId${F}1${F}10${F}5${F}20d${F}index_map"
    val config3InStr = s"randomEffectType${F}featureShardId${F}1${F}10${F}5${F}20d${F}identity"

    val argValueInStr = s"random1:$config1InStr|random2:$config2InStr|random3:$config3InStr"
    val params =
      GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(RANDOM_EFFECT_DATA_CONFIGURATIONS, argValueInStr)))
    val config1 = RandomEffectDataConfiguration.parseAndBuildFromString(config1InStr)
    val config2 = RandomEffectDataConfiguration.parseAndBuildFromString(config2InStr)
    val config3 = RandomEffectDataConfiguration.parseAndBuildFromString(config3InStr)
    val expectedValue = Map("random1" -> config1, "random2" -> config2, "random3" -> config3)
    assertEquals(params.randomEffectDataConfigurations, expectedValue)
  }

  @Test
  def testComputeVariance(): Unit = {
    val paramsAll = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(COMPUTE_VARIANCE, "trUE")))
    assertEquals(paramsAll.computeVariance, true)
    val paramsNone = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(COMPUTE_VARIANCE, "fAlSe")))
    assertEquals(paramsNone.computeVariance, false)
  }

  @Test
  def testSaveModelsToHDFS(): Unit = {
    val paramsAll = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(SAVE_MODELS_TO_HDFS, "true")))
    assertEquals(paramsAll.modelOutputMode, ModelOutputMode.ALL)
    val paramsNone = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(SAVE_MODELS_TO_HDFS, "FALSE")))
    assertEquals(paramsNone.modelOutputMode, ModelOutputMode.NONE)
  }

  @Test
  def testOutputModelModel(): Unit = {
    val paramsAll = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(MODEL_OUTPUT_MODE, "aLl")))
    assertEquals(paramsAll.modelOutputMode, ModelOutputMode.ALL)
    val paramsNone = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(MODEL_OUTPUT_MODE, "NoNe")))
    assertEquals(paramsNone.modelOutputMode, ModelOutputMode.NONE)
    val paramsBest = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(MODEL_OUTPUT_MODE, "bESt")))
    assertEquals(paramsBest.modelOutputMode, ModelOutputMode.BEST)
  }

  @Test
  def testNumOutputFilesForRandomEffectModel(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(NUM_OUTPUT_FILES_FOR_RANDOM_EFFECT_MODEL, "12")))
    assertEquals(params.numberOfOutputFilesForRandomEffectModel, 12)
  }

  @Test
  def testDeleteOutputDirIfExists(): Unit = {
    val paramsDeleteTrue = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(DELETE_OUTPUT_DIR_IF_EXISTS, "trUe")))
    assertEquals(paramsDeleteTrue.deleteOutputDirIfExists, true)
    val paramsDeleteFalse = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(DELETE_OUTPUT_DIR_IF_EXISTS, "faLSE")))
    assertEquals(paramsDeleteFalse.deleteOutputDirIfExists, false)
  }

  @Test
  def testApplicationName(): Unit = {
    val params = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(APPLICATION_NAME, "GAME_TEST")))
    assertEquals(params.applicationName, "GAME_TEST")
  }

  @Test
  def testOutputDir(): Unit = {
    // When output directory contains ':'
    val paramsWithColonAsPartOfOutputDir = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(OUTPUT_DIR, "hdfs://foo/bar/tar")))
    assertEquals(paramsWithColonAsPartOfOutputDir.outputDir, "hdfs://foo/bar/tar")

    // When output directory contains ',', the current logic will replace ',' with '_'
    val paramsWithCommaAsPartOfOutputDir = GameParams.parseFromCommandLine(mapToArray(setOneMoreArg(OUTPUT_DIR, "linkedin,airbnb")))
    assertEquals(paramsWithCommaAsPartOfOutputDir.outputDir, "linkedin_airbnb")
  }
}


object GameParamsTest {
  val defaultParams = new GameParams

  // Required parameters
  val TRAIN_INPUT_DIRS = "train-input-dirs"
  val TASK_TYPE = "task-type"
  val OUTPUT_DIR = "output-dir"
  val FEATURE_NAME_AND_TERM_SET_PATH = "feature-name-and-term-set-path"
  val UPDATING_SEQUENCE = "updating-sequence"

  // Optional parameters
  val TRAIN_DATE_RANGE = "train-date-range"
  val TRAIN_DATE_RANGE_DAYS_AGO = "train-date-range-days-ago"
  val VALIDATE_INPUT_DIRS = "validate-input-dirs"
  val VALIDATE_DATE_RANGE = "validate-date-range"
  val VALIDATE_DATE_RANGE_DAYS_AGO = "validate-date-range-days-ago"
  val MIN_PARTITIONS_FOR_VALIDATION = "min-partitions-for-validation"
  val FEATURE_SHARD_ID_TO_FEATURE_SECTION_KEYS_MAP = "feature-shard-id-to-feature-section-keys-map"
  val FEATURE_SHARD_ID_TO_INTERCEPT_MAP = "feature-shard-id-to-intercept-map"
  val NUM_ITERATIONS = "num-iterations"
  val COMPUTE_VARIANCE = "compute-variance"
  val FIXED_EFFECT_OPTIMIZATION_CONFIGURATIONS = "fixed-effect-optimization-configurations"
  val FIXED_EFFECT_DATA_CONFIGURATIONS = "fixed-effect-data-configurations"
  val RANDOM_EFFECT_OPTIMIZATION_CONFIGURATIONS = "random-effect-optimization-configurations"
  val FACTORED_RANDOM_EFFECT_OPTIMIZATION_CONFIGURATIONS = "factored-random-effect-optimization-configurations"
  val RANDOM_EFFECT_DATA_CONFIGURATIONS = "random-effect-data-configurations"
  val SAVE_MODELS_TO_HDFS = "save-models-to-hdfs"
  val MODEL_OUTPUT_MODE = "model-output-mode"
  val NUM_OUTPUT_FILES_FOR_RANDOM_EFFECT_MODEL = "num-output-files-for-random-effect-model"
  val DELETE_OUTPUT_DIR_IF_EXISTS = "delete-output-dir-if-exists"
  val APPLICATION_NAME = "application-name"

  val REQUIRED_OPTIONS =
    Array(TRAIN_INPUT_DIRS, OUTPUT_DIR, TASK_TYPE, FEATURE_NAME_AND_TERM_SET_PATH, UPDATING_SEQUENCE)

  // Get all required arguments
  def requiredArgs(): Map[String, String] = {
    val args = new Array[(String, String)](REQUIRED_OPTIONS.length)
    var i = 0
    REQUIRED_OPTIONS.foreach { option =>
      val name = fromOptionNameToArg(option)
      val value = option match {
        case TASK_TYPE => TaskType.LINEAR_REGRESSION.toString
        // We don't really care what the value is, as long as it meets the type requirement. But here we use the
        // option string itself as value, so that at least we can guarantee different names have different values.
        case _ => option
      }
      args(i) = (name, value)
      i += 1
    }
    args.toMap
  }

  // Get all required arguments except the one with name missingArgName
  def requiredArgsMissingOne(missingArgName: String): Map[String, String] = {
    if (REQUIRED_OPTIONS.isEmpty) {
      throw new RuntimeException("No required option configured in test.")
    }
    val args = new Array[(String, String)](REQUIRED_OPTIONS.length - 1)
    var i = 0
    REQUIRED_OPTIONS.filter(_ != missingArgName).foreach { option =>
      val name = fromOptionNameToArg(option)
      val value = option match {
        case TASK_TYPE => TaskType.LINEAR_REGRESSION.toString
        case _ => option
      }
      args(i) = (name, value)
      i += 1
    }
    args.toMap
  }

  // Set one more optional argument besides the required arguments
  def setOneMoreArg(argName: String, argValue: String): Map[String, String] = {
    val name = fromOptionNameToArg(argName)
    val value = argValue
    requiredArgs().updated(name, value)
  }
}
