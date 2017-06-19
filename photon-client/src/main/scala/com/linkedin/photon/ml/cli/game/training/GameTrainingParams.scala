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
package com.linkedin.photon.ml.cli.game.training

import scopt.OptionParser

import com.linkedin.photon.ml.HyperparameterTuningMode
import com.linkedin.photon.ml.HyperparameterTuningMode.HyperparameterTuningMode
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types._
import com.linkedin.photon.ml.cli.game.{EvaluatorParams, FeatureParams}
import com.linkedin.photon.ml.data.{InputColumnsNames, FixedEffectDataConfiguration, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.io.deprecated.ModelOutputMode
import com.linkedin.photon.ml.io.deprecated.ModelOutputMode.ModelOutputMode
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.game._
import com.linkedin.photon.ml.util.{PalDBIndexMapParams, Utils}
import com.linkedin.photon.ml.TaskType

/**
 * A bean class for GAME training parameters to replace the original case class for input parameters.
 *
 * TODO: This class needs a checkInvariant, located here, rather than having tests spread out all over.
 * TODO: Making the way GAME being configured more user friendly
 *
 * @note Note that examples of how to configure GAME parameters can be found in the integration tests for the GAME
 *       driver.
 */
class GameTrainingParams extends FeatureParams with PalDBIndexMapParams with EvaluatorParams {

  /**
   * Input directories of training data. Multiple input directories are also accepted if they are
   * separated by commas, e.g., inputDir1,inputDir2,inputDir3.
   */
  var trainDirs: Array[String] = Array()

  /**
   * Date range for the training data represented in the form start.date-end.date, e.g. 20150501-20150631.
   * If trainDateRangeOpt is specified, the input directory is expected to be in the daily format structure
   * (e.g., trainDir/daily/2015/05/20/input-data-files). Otherwise, the input paths are assumed to be flat
   * directories of input files (e.g., trainDir/input-data-files).
   */
  var trainDateRangeOpt: Option[String] = None

  /**
   * Date range for the training data represented in the form start.daysAgo-end.daysAgo, e.g. 90-1.
   * If trainDateRangeDaysAgoOpt is specified, the input directory is expected to be in the daily format structure
   * (e.g., trainDir/daily/2015/05/20/input-data-files).  Otherwise, the input paths are assumed to be flat directories
   * of input files (e.g., trainDir/input-data-files).
   */
  var trainDateRangeDaysAgoOpt: Option[String] = None

  /**
   * Input directories of validation data. Multiple input directories are also accepted if they are separated by
   * commas, e.g., inputDir1,inputDir2,inputDir3.
   */
  var validationDirsOpt: Option[Array[String]] = None

  /**
   * Date range for the training data represented in the form start.date-end.date, e.g. 20150501-20150631.
   * If validationDateRangeOpt is specified, the input directory is expected to be in the daily format structure
   * (e.g., validationDir/daily/2015/05/20/input-data-files). Otherwise, the input paths are assumed to be flat
   * directories of input files (e.g., validationDir/input-data-files)."
   */
  var validationDateRangeOpt: Option[String] = None

  /**
   * Date range for the training data represented in the form start.daysAgo-end.daysAgo, e.g. 90-1.
   * If validationDateRangeDaysAgoOpt is specified, the input directory is expected to be in the daily format structure
   * (e.g., validationDir/daily/2015/05/20/input-data-files). Otherwise, the input paths are assumed to be flat
   * directories of input files (e.g., validationDir/input-data-files).
   */
  var validationDateRangeDaysAgoOpt: Option[String] = None

  /**
   * If summarization output dir is provided, basic statistics of training data will be written to the given directory
   */
  var summarizationOutputDirOpt: Option[String] = None

  /**
   * Minimum number of partitions for validation data (if provided)
   */
  var minPartitionsForValidation: Int = 1

  /**
   * Output directory for logs and learned models
   */
  var outputDir: String = ""

  /**
   * Number of coordinate descent iterations, i.e. number of passes over all the coordinates. If set to 1, GAME
   * will do one pass over all the coordinates, optimizing each coordinate in turn, then stop.
   */
  var numIterations: Int = 1

  /**
   * Whether to compute coefficient variance
   */
  var computeVariance: Boolean = false

  /**
   * Updating order of the ordinates (separated by commas) in the coordinate descent algorithm
   */
  var updatingSequence: Seq[CoordinateId] = Seq()

  /**
   * Optimization configurations for the fixed effect optimization problem
   */
  var fixedEffectOptimizationConfigurations: Array[Map[CoordinateId, GLMOptimizationConfiguration]] = Array(Map())

  /**
   * Configurations for each fixed effect data set
   */
  var fixedEffectDataConfigurations: Map[CoordinateId, FixedEffectDataConfiguration] = Map()

  /**
   * Optimization configurations for each random effect optimization problem, multiple parameters are separated
   * by semi-colon.
   */
  var randomEffectOptimizationConfigurations: Array[Map[CoordinateId, GLMOptimizationConfiguration]] = Array(Map())

  /**
   * Optimization configurations for each factored random effect optimization problem, multiple parameters are
   * accepted and separated by semi-colon
   */
  var factoredRandomEffectOptimizationConfigurations:
    Array[Map[CoordinateId, FactoredRandomEffectOptimizationConfiguration]] = Array(Map())

  /**
   * Configurations for all the random effect data sets.
   */
  var randomEffectDataConfigurations: Map[CoordinateId, RandomEffectDataConfiguration] = Map()

  /**
   * GAME task type. Examples include logistic_regression and linear_regression
   */
  var taskType: TaskType = TaskType.LOGISTIC_REGRESSION

  /**
   * Model output mode (output all models, best model, or no models)
   */
  var modelOutputMode: ModelOutputMode = ModelOutputMode.ALL

  /**
   * Number of output files to write for each random effect model
   */
  var numberOfOutputFilesForRandomEffectModel: Int = -1

  /**
   * Whether to delete the output directory if exists
   */
  var deleteOutputDirIfExists: Boolean = false

  /**
   * Training data normalization method
   */
  var normalizationType: NormalizationType = NormalizationType.NONE

  /**
   * Name of this Spark application
   */
  var applicationName: String = "Game-Full-Model-Training"

  /**
   * Turn on additional tests on the training and validation data. Off by default, because it requires iterating
   * over the whole training/validation data set, which can be time consuming.
   */
  var checkData: Boolean = false

  /**
   * Names of the input columns to use when reading data
   */
  var inputColumnsNames = InputColumnsNames()

  /**
   * Whether to use warm start when optimizing multiple models with different regularization parameters. If warm start
   * is used, the previous optimal model is used to initialize the next model (for the next regularization parameter(s)
   * to use).
   */
  var useWarmStart: Boolean = false

  /**
   * Number of iterations to run hyperparameter tuning
   */
  var hyperparameterTuningIterations: Int = 0

  /**
   * Hyperparameter tuning mode
   */
  var hyperparameterTuningMode: HyperparameterTuningMode = HyperparameterTuningMode.BAYESIAN

  /**
   * Range of valid values for regularization weights
   */
  var regularizationWeightRange: (Double, Double) = (1e-4, 1e4)

  /**
   * Return all the model configs in one Array.
   */
  def getAllModelConfigs: Array[GameModelOptimizationConfiguration] = {
    for (
      fixedEffectOptimizationConfiguration <- fixedEffectOptimizationConfigurations;
      randomEffectOptimizationConfiguration <- randomEffectOptimizationConfigurations;
      factoredRandomEffectOptimizationConfiguration <- factoredRandomEffectOptimizationConfigurations) yield {

      GameModelOptimizationConfiguration(
        fixedEffectOptimizationConfiguration,
        randomEffectOptimizationConfiguration,
        factoredRandomEffectOptimizationConfiguration)
    }
  }

  override def toString: String = List(
    s"trainDirs: ${trainDirs.mkString(", ")}",
    s"trainDateRangeOpt: $trainDateRangeOpt",
    s"trainDateRangeDaysAgoOpt: $trainDateRangeDaysAgoOpt",
    s"validationDirsOpt: ${validationDirsOpt.map(_.mkString(", "))}",
    s"validationDateRangeOpt: $validationDateRangeOpt",
    s"validationDateRangeDaysAgoOpt: $validationDateRangeDaysAgoOpt",
    s"minNumPartitionsForValidation: $minPartitionsForValidation",
    s"featureNameAndTermSetInputPath: $featureNameAndTermSetInputPath",
    s"featureShardIdToFeatureSectionKeysMap:",
    s"${featureShardIdToFeatureSectionKeysMap.mapValues(_.mkString(", ")).mkString("\n")}",
    s"featureShardIdToInterceptMap:",
      s"${featureShardIdToInterceptMap.mkString("\n")}",
    s"outputDir: $outputDir",
    s"numIterations: $numIterations",
    s"updatingSequence: $updatingSequence",
    s"fixedEffectOptimizationConfigurations:",
      s"${fixedEffectOptimizationConfigurations.map(_.mkString("\n")).mkString("\n")}",
    s"fixedEffectDataConfigurations:",
      s"${fixedEffectDataConfigurations.mkString("\n")}",
    s"randomEffectOptimizationConfigurations:",
      s"${randomEffectOptimizationConfigurations.map(_.mkString("\n")).mkString("\n")}",
    s"factorRandomEffectOptimizationConfigurations:",
      s"${factoredRandomEffectOptimizationConfigurations.map(_.mkString("\n")).mkString("\n")}",
    s"randomEffectDataConfigurations:\n${randomEffectDataConfigurations.mkString("\n")}",
    s"taskType: $taskType",
    s"computeVariance: $computeVariance",
    s"modelOutputOption: $modelOutputMode",
    s"numberOfOutputFilesForRandomEffectModel: $numberOfOutputFilesForRandomEffectModel",
    s"deleteOutputDirIfExists: $deleteOutputDirIfExists",
    s"applicationName: $applicationName",
    s"offHeapIndexMapDir: $offHeapIndexMapDir",
    s"offHeapIndexMapNumPartitions: $offHeapIndexMapNumPartitions",
    s"normalizationType: $normalizationType",
    s"summarizationOutputDirOpt: $summarizationOutputDirOpt",
    s"checkData: $checkData",
    s"inputColumnsNames: $inputColumnsNames",
    s"useWarmStart: $useWarmStart",
    s"hyperparameterTuningIterations: $hyperparameterTuningIterations",
    s"hyperparameterTuningMode: $hyperparameterTuningMode",
    s"regularizationWeightRange: $regularizationWeightRange"
  ).mkString("\n")
}

object GameTrainingParams {

  // NOTE these are the parameter names for GAME, and Photon has its own, slightly different list.

  // Required parameters
  val TRAIN_INPUT_DIRS = "train-input-dirs"
  val TASK_TYPE = "task-type"
  val OUTPUT_DIR = "output-dir"
  val FEATURE_NAME_AND_TERM_SET_PATH = "feature-name-and-term-set-path"
  val UPDATING_SEQUENCE = "updating-sequence"

  // Optional parameters
  val TRAIN_DATE_RANGE = "train-date-range"
  val TRAIN_DATE_RANGE_DAYS_AGO = "train-date-range-days-ago"
  val VALIDATION_INPUT_DIRS = "validate-input-dirs"
  val VALIDATION_DATE_RANGE = "validate-date-range"
  val VALIDATION_DATE_RANGE_DAYS_AGO = "validate-date-range-days-ago"
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
  val EVALUATOR_TYPE = "evaluator-type"
  val SUMMARIZATION_OUTPUT_DIR = "summarization-output-dir"
  val NORMALIZATION_TYPE = "normalization-type"
  val OFFHEAP_INDEXMAP_DIR = "offheap-indexmap-dir"
  val OFFHEAP_INDEXMAP_NUM_PARTITIONS = "offheap-indexmap-num-partitions"
  val CHECK_DATA = "check-data"
  val INPUT_COLUMNS_NAMES = "input-column-names"
  val USE_WARM_START = "use-warm-start"
  val HYPERPARAMETER_TUNING_ITERATIONS = "hyperparameter-tuning-iterations"
  val HYPERPARAMETER_TUNING_MODE = "hyperparameter-tuning-mode"
  val REGULARIZATION_WEIGHT_RANGE = "regularization-weight-range"

  val defaultParams = new GameTrainingParams()

  /**
   * Parse parameters for GAME from the arguments on the command line.
   *
   * @param args An array containing each command line argument
   * @return An instance of [[GameTrainingParams]] or an exception if the parameters cannot be parsed correctly
   */
  def parseFromCommandLine(args: Array[String]): GameTrainingParams = {

    val params = new GameTrainingParams()
    val parser = new OptionParser[Unit]("Photon-Game") {

      opt[String](TRAIN_INPUT_DIRS)
        .required()
        .text("Input directories of training data. Multiple input directories are also accepted if they are " +
          "separated by commas, e.g., inputDir1,inputDir2,inputDir3.")
        .foreach(x => params.trainDirs = x.split(","))

      opt[String](TASK_TYPE)
        .required()
        .text("Task type. Examples include logistic_regression and linear_regression.")
        .foreach(x => params.taskType = TaskType.withName(x.toUpperCase))

      opt[String](OUTPUT_DIR)
        .required()
        .text(s"Output directory for logs and learned models.")
        .foreach(x => params.outputDir = x.replace(',', '_'))

      opt[String](FEATURE_NAME_AND_TERM_SET_PATH)
        .required()
        .text(s"Input path to the features name-and-term lists.\n" +
          s"DEPRECATED -- This option will be removed in the next major version. Use the offheap index map " +
          s"configuration instead")
        .foreach(x => params.featureNameAndTermSetInputPath = x)

      opt[String](UPDATING_SEQUENCE)
        .required()
        .text(s"Updating order of the ordinates (separated by commas) in the coordinate descent algorithm. It is " +
          s"recommended to order different fixed/random effect models based on their stability (e.g., by looking " +
          s"at the variance of the feature distribution (or correlation with labels) for each of the " +
          s"fixed/random effect model),")
        .foreach(x => params.updatingSequence = x.split(","))

      opt[String](TRAIN_DATE_RANGE)
        .text(s"Date range for the training data represented in the form start.date-end.date, " +
          s"e.g. 20150501-20150631. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., trainDir/daily/2015/05/20/input-data-files). Otherwise, the input paths" +
          s" are assumed to be flat directories of input files (e.g., trainDir/input-data-files). " +
          s"Default: ${defaultParams.trainDateRangeOpt}.")
        .foreach(x => params.trainDateRangeOpt = Some(x))

      opt[String](TRAIN_DATE_RANGE_DAYS_AGO)
        .text(s"Date range for the training data represented in the form start.daysAgo-end.daysAgo, " +
          s"e.g. 90-1. If this parameter is specified, the input directory is expected to be in the daily " +
          s"format structure (e.g., trainDir/daily/2015/05/20/input-data-files). Otherwise, the input paths " +
          s"are assumed to be flat directories of input files (e.g., trainDir/input-data-files). " +
          s"Default: ${defaultParams.trainDateRangeDaysAgoOpt}.")
        .foreach(x => params.trainDateRangeDaysAgoOpt = Some(x))

      opt[String](VALIDATION_INPUT_DIRS)
        .text("Input directories of validation data. Multiple input directories are also accepted if they are " +
          "separated by commas, e.g., inputDir1,inputDir2,inputDir3.")
        .foreach(x => params.validationDirsOpt = Some(x.split(",")))

      opt[String](VALIDATION_DATE_RANGE)
        .text(s"Date range for the validation data represented in the form start.date-end.date, " +
          s"e.g. 20150501-20150631. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., validationDir/daily/2015/05/20/input-data-files). Otherwise, the input " +
          s"paths are assumed to be flat directories of input files (e.g., validationDir/input-data-files). " +
          s"Default: ${defaultParams.validationDateRangeOpt}.")
        .foreach(x => params.validationDateRangeOpt = Some(x))

      opt[String](VALIDATION_DATE_RANGE_DAYS_AGO)
        .text(s"Date range for the validation data represented in the form start.daysAgo-end.daysAgo, " +
          s"e.g. 90-1. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., validationDir/daily/2015/05/20/input-data-files). Otherwise, the input " +
          s"paths are assumed to be flat directories of input files (e.g., validationDir/input-data-files). " +
          s"Default: ${defaultParams.validationDateRangeDaysAgoOpt}.")
        .foreach(x => params.validationDateRangeDaysAgoOpt = Some(x))

      opt[Int](MIN_PARTITIONS_FOR_VALIDATION)
        .text(s"Minimum number of partitions for validation data (if provided). " +
          s"Default: ${defaultParams.minPartitionsForValidation}")
        .foreach(x => params.minPartitionsForValidation = x)

      opt[String](FEATURE_SHARD_ID_TO_FEATURE_SECTION_KEYS_MAP)
        .text(s"A map between the feature shard id and it's corresponding feature section keys, in the following " +
          s"format: shardId1:sectionKey1,sectionKey2|shardId2:sectionKey2,sectionKey3.")
        .foreach(x => params.featureShardIdToFeatureSectionKeysMap =
          x.split("\\|")
            .map { line =>
              line.split(":") match {
                case Array(key, names) => (key, names.split(",").map(_.trim).toSet)
                case Array(key) => (key, Set[String]())
              }
            }
            .toMap
        )

      opt[String](FEATURE_SHARD_ID_TO_INTERCEPT_MAP)
        .text(s"A map between the feature shard id and a boolean variable that decides whether a dummy feature " +
          s"should be added to the corresponding shard in order to learn an intercept, for example, in the " +
          s"following format: shardId1:true|shardId2:false. The default is true for all shard ids.")
        .foreach(x => params.featureShardIdToInterceptMap =
          x.split("\\|")
            .map { line => line.split(":") match {
              case Array(key, flag) => (key, flag.toBoolean)
              case Array(key) => (key, true)
            }}
            .toMap
        )

      opt[Int](NUM_ITERATIONS)
        .text(s"Number of coordinate descent iterations, default: ${defaultParams.numIterations}")
        .foreach(x => params.numIterations = x)

      opt[String](FIXED_EFFECT_OPTIMIZATION_CONFIGURATIONS)
        .text("Optimization configurations for the fixed effect optimization problem. " +
          s"Expected format (if the GAME model contains two fixed effect model called model1 and mode2):\n" +
          s"model1:${GLMOptimizationConfiguration.EXPECTED_FORMAT}" +
          s"|model2:${GLMOptimizationConfiguration.EXPECTED_FORMAT}.\nMultiple configurations are " +
          "accepted and should be separated by semi-colon \";\".")
        .foreach(x => params.fixedEffectOptimizationConfigurations =
          x.split(";")
            .map(_.split("\\|")
              .map { line =>
                val Array(key, value) = line.split(":").map(_.trim)
                (key, GLMOptimizationConfiguration.parseAndBuildFromString(value))
              }
              .toMap)
        )

      opt[String](FIXED_EFFECT_DATA_CONFIGURATIONS)
        .text("Configurations for each fixed effect data set.")
        .foreach(x => params.fixedEffectDataConfigurations =
          x.split("\\|")
            .map { line =>
              val Array(key, value) = line.split(":").map(_.trim)
              (key, FixedEffectDataConfiguration.parseAndBuildFromString(value))
            }
            .toMap
        )

      opt[String](RANDOM_EFFECT_OPTIMIZATION_CONFIGURATIONS)
        .text("Optimization configurations for each random effect optimization problem. " +
          s"Expected format (if the GAME model contains two random effect model called model1 and mode2):\n" +
          s"model1:${GLMOptimizationConfiguration.EXPECTED_FORMAT}" +
          s"|model2:${GLMOptimizationConfiguration.EXPECTED_FORMAT}.\nMultiple configurations are " +
          "accepted and should be separated by semi-colon \";\".")
        .foreach(x => params.randomEffectOptimizationConfigurations =
          x.split(";")
            .map(_.split("\\|")
              .map { line =>
                val Array(key, value) = line.split(":").map(_.trim)
                (key, GLMOptimizationConfiguration.parseAndBuildFromString(value))
              }
              .toMap)
        )

      opt[String](FACTORED_RANDOM_EFFECT_OPTIMIZATION_CONFIGURATIONS)
        .text("Optimization configurations for each factored random effect optimization problem, multiple " +
          "parameters are accepted and separated by semi-colon \";\".")
        .foreach(x => params.factoredRandomEffectOptimizationConfigurations =
          x.split(";")
            .map(_.split("\\|")
              .map { line =>
                val Array(key, s1, s2, s3) = line.split(":").map(_.trim)
                val randomEffectOptConfig = GLMOptimizationConfiguration.parseAndBuildFromString(s1)
                val latentFactorOptConfig = GLMOptimizationConfiguration.parseAndBuildFromString(s2)
                val mfOptimizationOptConfig = MFOptimizationConfiguration.parseAndBuildFromString(s3)
                (key, FactoredRandomEffectOptimizationConfiguration(
                  randomEffectOptConfig, latentFactorOptConfig, mfOptimizationOptConfig))
              }
              .toMap)
        )

      opt[String](RANDOM_EFFECT_DATA_CONFIGURATIONS)
        .text("Configurations for all the random effect data sets.")
        .foreach(x => params.randomEffectDataConfigurations =
          x.split("\\|")
            .map { line =>
              val Array(key, value) = line.split(":").map(_.trim)
              (key, RandomEffectDataConfiguration.parseAndBuildFromString(value))
            }
            .toMap
        )

      opt[Boolean](COMPUTE_VARIANCE)
        .text(s"Whether to compute the coefficient variance, default: ${defaultParams.computeVariance}")
        .foreach(x => params.computeVariance = x)

      opt[Boolean](SAVE_MODELS_TO_HDFS)
        .text(s"DEPRECATED -- USE model-output-mode")
        .foreach(x => params.modelOutputMode = if (x) ModelOutputMode.ALL else ModelOutputMode.NONE)

      opt[String](MODEL_OUTPUT_MODE)
        .text(s"Output mode of trained models to HDFS (ALL, BEST, or NONE). Default: ${defaultParams.modelOutputMode}")
        .foreach(x => params.modelOutputMode = ModelOutputMode.withName(x.toUpperCase))

      opt[Int](NUM_OUTPUT_FILES_FOR_RANDOM_EFFECT_MODEL)
        .text(s"Number of output files to write for each random effect model. Not setting this parameter or " +
          s"setting it to -1 means to use the default number of output files." +
          s"Default: ${defaultParams.numberOfOutputFilesForRandomEffectModel}")
        .foreach(x => params.numberOfOutputFilesForRandomEffectModel = x)

      opt[Boolean](DELETE_OUTPUT_DIR_IF_EXISTS)
        .text(s"Whether to delete the output directory if exists. Default: ${defaultParams.deleteOutputDirIfExists}")
        .foreach(x => params.deleteOutputDirIfExists = x)

      opt[String]("application-name")
        .text(s"Name of this Spark application. Default: ${defaultParams.applicationName}.")
        .foreach(x => params.applicationName = x)

      opt[String](OFFHEAP_INDEXMAP_DIR)
        .text("The offheap storage directory if offheap map is needed. DefaultIndexMap will be used if not specified.")
        .foreach(x => params.offHeapIndexMapDir = Some(x))

      opt[Int](OFFHEAP_INDEXMAP_NUM_PARTITIONS)
        .text("The number of partitions for the offheap map storage. This partition number should be consistent with " +
            "the number when offheap storage is built. This parameter affects only the execution speed during " +
            "feature index building and has zero performance impact on training other than maintaining a " +
            "convention.")
        .foreach(x => params.offHeapIndexMapNumPartitions = x)

      opt[String](EVALUATOR_TYPE)
        .text("Type of the evaluator used to evaluate the computed scores.")
        .foreach(x => params.evaluatorTypes = Some(x.split(",").map(Utils.evaluatorWithName)))

      opt[String](NORMALIZATION_TYPE)
        .text("The normalization type to use in training. Options: " +
          s"[${NormalizationType.values().mkString("|")}]. Default: ${defaultParams.normalizationType}")
        .foreach(x => params.normalizationType = NormalizationType.valueOf(x.toUpperCase))

      opt[String](SUMMARIZATION_OUTPUT_DIR)
        .text("An optional directory to output statistics about the training data")
        .foreach(x => params.summarizationOutputDirOpt = Some(x))

      opt[Boolean](CHECK_DATA)
        .text(s"Whether to check the data or not. " +
          s"default: ${defaultParams.checkData}")
        .foreach(x => params.checkData = x)

      opt[String](INPUT_COLUMNS_NAMES)
        .text(s"A map of the input columns names, in format: response:label|offset:offset|uid:uid ... ")
        .foreach { x =>
          x.split("\\|").foreach { line =>
            line.split(":").map(_.trim) match {
              case Array(key, value) => params.inputColumnsNames.updated(InputColumnsNames.withName(key), value)
              case other => s"Whoops, couldn't interpret: '$other'"
            }
          }
        }

      opt[Boolean](USE_WARM_START)
        .text(s"Whether to use warm start in hyper-parameter tuning. Default: ${defaultParams.useWarmStart}")
        .foreach(x => params.useWarmStart)

      opt[Int](HYPERPARAMETER_TUNING_ITERATIONS)
        .text("The maximum number of iterations (i.e. model training / evaluation cycles) for hyperparameter tuning")
        .foreach(x => params.hyperparameterTuningIterations = x)

      opt[String](HYPERPARAMETER_TUNING_MODE)
        .text("Hyperparameter tuning mode. Options: " +
          s"[${HyperparameterTuningMode.values.mkString("|")}]. Default: ${defaultParams.hyperparameterTuningMode}")
        .foreach(x => params.hyperparameterTuningMode = HyperparameterTuningMode.withName(x.toUpperCase))

      opt[String](REGULARIZATION_WEIGHT_RANGE)
        .text("Range of valid values for regularization weights, separated by comma (e.g. 1e-4,1e4).")
        .foreach { x =>
          x.split(",") match {
            case Array(lower, upper) => params.regularizationWeightRange = (lower.toDouble, upper.toDouble)
            case other =>
              throw new IllegalArgumentException(
                s"Couldn't interpret $other as a range. Please specify the range with lower and upper bounds " +
                  s"separated by comma (e.g.: 1e-4,1e4).")
          }
        }

      help("help").text("prints usage text.")

      override def showUsageOnError = true
    }

    if (!parser.parse(args)) {

      val argsString =
        (for (i <- 0 until args.length/2 by 2)
          yield args(2*i) + ": " + args(2*i+1))
        .mkString("\n")

      throw new IllegalArgumentException(s"Parsing the command line arguments failed.\n" +
        s"Input arguments are:\n$argsString")
    }

    params
  }
}
