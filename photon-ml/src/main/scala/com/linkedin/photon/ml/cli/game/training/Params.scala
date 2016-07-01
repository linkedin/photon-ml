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
package com.linkedin.photon.ml.cli.game.training

import com.linkedin.photon.ml.optimization.game.{MFOptimizationConfiguration, GLMOptimizationConfiguration}
import com.linkedin.photon.ml.data.{FixedEffectDataConfiguration, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.io.ModelOutputMode
import com.linkedin.photon.ml.io.ModelOutputMode._
import com.linkedin.photon.ml.supervised.TaskType
import com.linkedin.photon.ml.supervised.TaskType._
import scopt.OptionParser

import scala.collection.{Map, Set}

/**
 * A bean class for GAME training parameters to replace the original case class for input parameters.
 * @note Note that examples of how to configure GAME parameters can be found in the integration tests for the GAME
 *       driver.
 * @todo Making the way GAME being configured more user friendly
 */
class Params {

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
   * Input directories of validating data. Multiple input directories are also accepted if they are separated by
   * commas, e.g., inputDir1,inputDir2,inputDir3.
   */
  var validateDirsOpt: Option[Array[String]] = None

  /**
   * Date range for the training data represented in the form start.date-end.date, e.g. 20150501-20150631.
   * If validateDateRangeOpt is specified, the input directory is expected to be in the daily format structure
   * (e.g., validateDir/daily/2015/05/20/input-data-files). Otherwise, the input paths are assumed to be flat
   * directories of input files (e.g., validateDir/input-data-files)."
   */
  var validateDateRangeOpt: Option[String] = None

  /**
   * Date range for the training data represented in the form start.daysAgo-end.daysAgo, e.g. 90-1.
   * If validateDateRangeDaysAgoOpt is specified, the input directory is expected to be in the daily format structure
   * (e.g., validateDir/daily/2015/05/20/input-data-files). Otherwise, the input paths are assumed to be flat
   * directories of input files (e.g., validateDir/input-data-files).
   */
  var validateDateRangeDaysAgoOpt: Option[String] = None

  /**
   * Minimum number of partitions for validating data (if provided).
   */
  var minPartitionsForValidation: Int = 1

  /**
   * Input path to the features name-and-term lists.
   */
  var featureNameAndTermSetInputPath: String = ""

  /**
   * A map between the feature shard id and it's corresponding feature section keys in the following format:
   * shardId1:sectionKey1,sectionKey2|shardId2:sectionKey2,sectionKey3.
   */
  var featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]] = Map()

  /**
   * A map between the feature shard id and a boolean variable that decides whether a dummy feature should be added
   * to the corresponding shard in order to learn an intercept, for example,
   * in the following format: shardId1:true|shardId2:false. The default is true for all or unspecified shard ids.
   */
  var featureShardIdToInterceptMap: Map[String, Boolean] = Map()

  /**
   * Output directory for logs and learned models.
   */
  var outputDir: String = ""

  /**
   * Number of coordinate descent iterations.
   */
  var numIterations: Int = 1

  /**
   * Updating order of the ordinates (separated by commas) in the coordinate descent algorithm.
   */
  var updatingSequence: Seq[String] = Seq()

  /**
   * Optimization configurations for the fixed effect optimization problem.
   */
  var fixedEffectOptimizationConfigurations: Array[Map[String, GLMOptimizationConfiguration]] = Array(Map())

  /**
   * Configurations for each fixed effect data set.
   */
  var fixedEffectDataConfigurations: Map[String, FixedEffectDataConfiguration] = Map()

  /**
   * Optimization configurations for each random effect optimization problem, multiple parameters are separated
   * by semi-colon.
   */
  var randomEffectOptimizationConfigurations: Array[Map[String, GLMOptimizationConfiguration]] = Array(Map())

  /**
   * Optimization configurations for each factored random effect optimization problem, multiple parameters are
   * accepted and separated by semi-colon.
   */
  var factoredRandomEffectOptimizationConfigurations
    : Array[Map[String, (GLMOptimizationConfiguration, GLMOptimizationConfiguration, MFOptimizationConfiguration)]]
  = Array(Map())

  /**
   * Configurations for all the random effect data sets.
   */
  var randomEffectDataConfigurations: Map[String, RandomEffectDataConfiguration] = Map()

  /**
   * GAME task type. Examples include logistic_regression and linear_regression.
   */
  var taskType: TaskType = LOGISTIC_REGRESSION

  /**
   * Model output mode (output all models, best model, or no models)
   */
  var modelOutputMode: ModelOutputMode = ALL

  /**
   * Number of output files to write for each random effect model.
   */
  var numberOfOutputFilesForRandomEffectModel: Int = -1

  /**
   * Whether to delete the output directory if exists
   */
  var deleteOutputDirIfExists: Boolean = false

  /**
   * Name of this Spark application
   */
  var applicationName: String = "Game-Full-Model-Training"

  override def toString: String = {
    s"trainDirs: ${trainDirs.mkString(", ")}\n" +
        s"trainDateRangeOpt: $trainDateRangeOpt\n" +
        s"trainDateRangeDaysAgoOpt: $trainDateRangeDaysAgoOpt\n" +
        s"validateDirsOpt: ${validateDirsOpt.map(_.mkString(", "))}\n" +
        s"validateDateRangeOpt: $validateDateRangeOpt\n" +
        s"validateDateRangeDaysAgoOpt: $validateDateRangeDaysAgoOpt\n" +
        s"minNumPartitionsForValidation: $minPartitionsForValidation\n" +
        s"featureNameAndTermSetInputPath: $featureNameAndTermSetInputPath\n" +
        s"featureShardIdToFeatureSectionKeysMap:\n${featureShardIdToFeatureSectionKeysMap.mapValues(_.mkString(", "))
            .mkString("\n")}\n" +
        s"featureShardIdToInterceptMap:\n${featureShardIdToInterceptMap.mkString("\n")}" +
        s"outputDir: $outputDir\n" +
        s"numIterations: $numIterations\n" +
        s"updatingSequence: $updatingSequence\n" +
        s"fixedEffectOptimizationConfigurations:\n${fixedEffectOptimizationConfigurations.map(_.mkString("\n"))
            .mkString("\n")}\n" +
        s"fixedEffectDataConfigurations: \n${fixedEffectDataConfigurations.mkString("\n")}\n" +
        s"randomEffectOptimizationConfigurations:\n${randomEffectOptimizationConfigurations.map(_.mkString("\n"))
            .mkString("\n")}\n" +
        s"factorRandomEffectOptimizationConfigurations:\n${factoredRandomEffectOptimizationConfigurations
            .map(_.mkString("\n")).mkString("\n")}\n" +
        s"randomEffectDataConfigurations:\n${randomEffectDataConfigurations.mkString("\n")}\n" +
        s"taskType: $taskType\n" +
        s"modelOutputOption: $modelOutputMode\n" +
        s"numberOfOutputFilesForRandomEffectModel: $numberOfOutputFilesForRandomEffectModel\n" +
        s"deleteOutputDirIfExists: $deleteOutputDirIfExists\n" +
        s"applicationName: $applicationName"
  }
}

object Params {
  protected[training] def parseFromCommandLine(args: Array[String]): Params = {
    val defaultParams = new Params()
    val params = new Params()
    val parser = new OptionParser[Unit]("Photon-Game") {
      opt[String]("train-input-dirs")
        .required()
        .text("Input directories of training data. Multiple input directories are also accepted if they are " +
          "separated by commas, e.g., inputDir1,inputDir2,inputDir3.")
        .foreach(x => params.trainDirs = x.split(","))
      opt[String]("task-type")
        .required()
        .text("Task type. Examples include logistic_regression and linear_regression.")
        .foreach(x => params.taskType = TaskType.withName(x.toUpperCase))
      opt[String]("output-dir")
        .required()
        .text(s"Output directory for logs and learned models.")
        .foreach(x => params.outputDir = x.replace(',', '_'))
      opt[String]("feature-name-and-term-set-path")
        .required()
        .text(s"Input path to the features name-and-term lists.")
        .foreach(x => params.featureNameAndTermSetInputPath = x)
      opt[String]("train-date-range")
        .text(s"Date range for the training data represented in the form start.date-end.date, " +
          s"e.g. 20150501-20150631. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., trainDir/daily/2015/05/20/input-data-files). Otherwise, the input paths" +
          s" are assumed to be flat directories of input files (e.g., trainDir/input-data-files). " +
          s"Default: ${defaultParams.trainDateRangeOpt}.")
        .foreach(x => params.trainDateRangeOpt = Some(x))
      opt[String]("train-date-range-days-ago")
        .text(s"Date range for the training data represented in the form start.daysAgo-end.daysAgo, " +
          s"e.g. 90-1. If this parameter is specified, the input directory is expected to be in the daily " +
          s"format structure (e.g., trainDir/daily/2015/05/20/input-data-files). Otherwise, the input paths " +
          s"are assumed to be flat directories of input files (e.g., trainDir/input-data-files). " +
          s"Default: ${defaultParams.trainDateRangeDaysAgoOpt}.")
        .foreach(x => params.trainDateRangeDaysAgoOpt = Some(x))
      opt[String]("validate-input-dirs")
        .text("Input directories of validating data. Multiple input directories are also accepted if they are " +
          "separated by commas, e.g., inputDir1,inputDir2,inputDir3.")
        .foreach(x => params.validateDirsOpt = Some(x.split(",")))
      opt[String]("validate-date-range")
        .text(s"Date range for the validating data represented in the form start.date-end.date, " +
          s"e.g. 20150501-20150631. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., validateDir/daily/2015/05/20/input-data-files). Otherwise, the input " +
          s"paths are assumed to be flat directories of input files (e.g., validateDir/input-data-files). " +
          s"Default: ${defaultParams.validateDateRangeOpt}.")
        .foreach(x => params.validateDateRangeOpt = Some(x))
      opt[String]("validate-date-range-days-ago")
        .text(s"Date range for the validating data represented in the form start.daysAgo-end.daysAgo, " +
          s"e.g. 90-1. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., validateDir/daily/2015/05/20/input-data-files). Otherwise, the input " +
          s"paths are assumed to be flat directories of input files (e.g., validateDir/input-data-files). " +
          s"Default: ${defaultParams.validateDateRangeDaysAgoOpt}.")
        .foreach(x => params.validateDateRangeDaysAgoOpt = Some(x))
      opt[Int]("min-partitions-for-validation")
        .text(s"Minimum number of partitions for validating data (if provided). " +
          s"Default: ${defaultParams.minPartitionsForValidation}")
        .foreach(x => params.minPartitionsForValidation = x)
      opt[String]("feature-shard-id-to-feature-section-keys-map")
        .text(s"A map between the feature shard id and it's corresponding feature section keys, in the following " +
          s"format: shardId1:sectionKey1,sectionKey2|shardId2:sectionKey2,sectionKey3.")
        .foreach(x => params.featureShardIdToFeatureSectionKeysMap =
          x.split("\\|")
            .map { line => line.split(":") match {
              case Array(key, names) => (key, names.split(",").map(_.trim).toSet)
              case Array(key) => (key, Set[String]())
            }}
            .toMap
        )
      opt[String]("feature-shard-id-to-intercept-map")
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
      opt[Int]("num-iterations")
        .text(s"Number of coordinate descent iterations, default: ${defaultParams.numIterations}")
        .foreach(x => params.numIterations = x)
      opt[String]("fixed-effect-optimization-configurations")
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
      opt[String]("fixed-effect-data-configurations")
        .text("Configurations for each fixed effect data set.")
        .foreach(x => params.fixedEffectDataConfigurations =
          x.split("\\|")
            .map { line =>
              val Array(key, value) = line.split(":").map(_.trim)
              (key, FixedEffectDataConfiguration.parseAndBuildFromString(value))
            }
            .toMap
        )
      opt[String]("updating-sequence")
        .text(s"Updating order of the ordinates (separated by commas) in the coordinate descent algorithm.")
        .foreach(x => params.updatingSequence = x.split(","))
      opt[String]("random-effect-optimization-configurations")
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
      opt[String]("factored-random-effect-optimization-configurations")
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
                (key, (randomEffectOptConfig, latentFactorOptConfig, mfOptimizationOptConfig))
              }
              .toMap)
        )
      opt[String]("random-effect-data-configurations")
        .text("Configurations for all the random effect data sets.")
        .foreach(x => params.randomEffectDataConfigurations =
          x.split("\\|")
            .map { line =>
              val Array(key, value) = line.split(":").map(_.trim)
              (key, RandomEffectDataConfiguration.parseAndBuildFromString(value))
            }
            .toMap
        )
      opt[Boolean]("save-models-to-hdfs")
        .text(s"DEPRECATED -- USE model-output-mode")
        .foreach(x => params.modelOutputMode = if (x) ALL else NONE)
      opt[String]("model-output-mode")
        .text(s"Output mode of trained models to HDFS (ALL, BEST, or NONE). Default: ${defaultParams.modelOutputMode}")
        .foreach(x => params.modelOutputMode = ModelOutputMode.withName(x.toUpperCase))
      opt[Int]("num-output-files-for-random-effect-model")
        .text(s"Number of output files to write for each random effect model. Not setting this parameter or " +
          s"setting it to -1 means to use the default number of output files." +
          s"Default: ${defaultParams.numberOfOutputFilesForRandomEffectModel}")
        .foreach(x => params.numberOfOutputFilesForRandomEffectModel = x)
      opt[Boolean]("delete-output-dir-if-exists")
        .text(s"Whether to delete the output directory if exists. Default: ${defaultParams.deleteOutputDirIfExists}")
        .foreach(x => params.deleteOutputDirIfExists = x)
      opt[String]("application-name")
        .text(s"Name of this Spark application. Default: ${defaultParams.applicationName}.")
        .foreach(x => params.applicationName = x)
      help("help").text("prints usage text.")
      override def showUsageOnError = true
    }
    if (!parser.parse(args)) {
      throw new IllegalArgumentException(s"Parsing the command line arguments failed.\n" +
        s"Input arguments are: ${args.mkString(", ")}).")
    }
    params
  }
}
