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
 * Command line arguments for GAME
 *
 * @param trainDirs Input directories of training data. Multiple input directories are also accepted if they are
 *                  separated by commas, e.g., inputDir1,inputDir2,inputDir3.
 * @param trainDateRangeOpt Date range for the training data represented in the form start.date-end.date,
 *                          e.g. 20150501-20150631. If trainDateRangeOpt is specified, the input directory is expected
 *                          to be in the daily format structure (e.g., trainDir/daily/2015/05/20/input-data-files).
 *                          Otherwise, the input paths are assumed to be flat directories of input files
 *                          (e.g., trainDir/input-data-files)."
 * @param trainDateRangeDaysAgoOpt Date range for the training data represented in the form start.daysAgo-end.daysAgo,
 *                          e.g. 90-1. If trainDateRangeDaysAgoOpt is specified, the input directory is expected
 *                          to be in the daily format structure (e.g., trainDir/daily/2015/05/20/input-data-files).
 *                          Otherwise, the input paths are assumed to be flat directories of input files
 *                          (e.g., trainDir/input-data-files)."
 * @param validateDirsOpt Input directories of validating data. Multiple input directories are also accepted if they
 *                        are separated by commas, e.g., inputDir1,inputDir2,inputDir3.
 * @param validateDateRangeOpt Date range for the training data represented in the form start.date-end.date,
 *                             e.g. 20150501-20150631. If validateDateRangeOpt is specified, the input directory is
 *                             expected to be in the daily format structure
 *                             (e.g., validateDir/daily/2015/05/20/input-data-files). Otherwise, the input paths are
 *                             assumed to be flat directories of input files (e.g., validateDir/input-data-files)."
 * @param validateDateRangeDaysAgoOpt Date range for the training data represented in the form
 *                                    start.daysAgo-end.daysAgo, e.g. 90-1. If validateDateRangeDaysAgoOpt is specified,
 *                                    the input directory is expected to be in the daily format structure (e.g.,
 *                                    validateDir/daily/2015/05/20/input-data-files). Otherwise, the input paths are
 *                                    assumed to be flat directories of input files (e.g.,
 *                                    validateDir/input-data-files)."
 * @param minPartitionsForValidation Minimum number of partitions for validating data (if provided).
 * @param featureNameAndTermSetInputPath Input path to the features name-and-term lists.
 * @param featureShardIdToFeatureSectionKeysMap A map between the feature shard id and it's corresponding feature
 *                                              section keys in the following format:
 *                                              shardId1:sectionKey1,sectionKey2|shardId2:sectionKey2,sectionKey3.
 * @param outputDir Output directory for logs and learned models.
 * @param numIterations Number of coordinate descent iterations.
 * @param updatingSequence Updating order of the ordinates (separated by commas) in the coordinate descent algorithm.
 * @param fixedEffectOptimizationConfigurations Optimization configurations for the fixed effect optimization problem,
 *                                              multiple configurations are accepted and should be separated by
 *                                              semi-colon.
 * @param fixedEffectDataConfigurations Configurations for each fixed effect data set.
 * @param randomEffectOptimizationConfigurations Optimization configurations for each random effect optimization
 *                                               problem, multiple parameters are separated by semi-colon.
 * @param factoredRandomEffectOptimizationConfigurations Optimization configurations for each factored random effect
 *                                                       optimization problem, multiple parameters are accepted and
 *                                                       separated by semi-colon.
 * @param randomEffectDataConfigurations Configurations for all the random effect data sets.
 * @param taskType GAME task type. Examples include logistic_regression and linear_regression.
 * @param modelOutputMode Model output mode (output all models, best model, or no models)
 * @param numberOfOutputFilesForRandomEffectModel Number of output files to write for each random effect model.
 * @param applicationName Name of this Spark application.
 * @note Note that examples of how to configure GAME parameters can be found in the integration tests for the GAME
 *       driver.
 * @todo Making the way GAME being configured more user friendly
 */
case class Params(
    trainDirs: Array[String] = Array(),
    trainDateRangeOpt: Option[String] = None,
    trainDateRangeDaysAgoOpt: Option[String] = None,
    validateDirsOpt: Option[Array[String]] = None,
    validateDateRangeOpt: Option[String] = None,
    validateDateRangeDaysAgoOpt: Option[String] = None,
    minPartitionsForValidation: Int = 1,
    featureNameAndTermSetInputPath: String = "",
    featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]] = Map(),
    outputDir: String = "",
    numIterations: Int = 1,
    updatingSequence: Seq[String] = Seq(),
    fixedEffectOptimizationConfigurations: Array[Map[String, GLMOptimizationConfiguration]] = Array(Map()),
    fixedEffectDataConfigurations: Map[String, FixedEffectDataConfiguration] = Map(),
    randomEffectOptimizationConfigurations: Array[Map[String, GLMOptimizationConfiguration]] = Array(Map()),
    factoredRandomEffectOptimizationConfigurations: Array[Map[String,
        (GLMOptimizationConfiguration, GLMOptimizationConfiguration, MFOptimizationConfiguration)]] = Array(Map()),
    randomEffectDataConfigurations: Map[String, RandomEffectDataConfiguration] = Map(),
    taskType: TaskType = LOGISTIC_REGRESSION,
    modelOutputMode: ModelOutputMode = ALL,
    numberOfOutputFilesForRandomEffectModel: Int = -1,
    applicationName: String = "Game-Full-Model-Training") {

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
      s"applicationName: $applicationName"
  }
}

object Params {
  protected[training] def parseFromCommandLine(args: Array[String]): Params = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Photon-Game") {
      opt[String]("train-input-dirs")
        .required()
        .text("Input directories of training data. Multiple input directories are also accepted if they are " +
          "separated by commas, e.g., inputDir1,inputDir2,inputDir3.")
        .action((x, c) => c.copy(trainDirs = x.split(",")))
      opt[String]("task-type")
        .required()
        .text("Task type. Examples include logistic_regression and linear_regression.")
        .action((x, c) => c.copy(taskType = TaskType.withName(x.toUpperCase)))
      opt[String]("train-date-range")
        .text(s"Date range for the training data represented in the form start.date-end.date, " +
          s"e.g. 20150501-20150631. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., trainDir/daily/2015/05/20/input-data-files). Otherwise, the input paths " +
          s"are assumed to be flat directories of input files (e.g., trainDir/input-data-files). " +
          s"Default: ${defaultParams.trainDateRangeOpt}.")
        .action((x, c) => c.copy(trainDateRangeOpt = Some(x)))
      opt[String]("train-date-range-days-ago")
        .text(s"Date range for the training data represented in the form start.daysAgo-end.daysAgo, " +
          s"e.g. 90-1. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., trainDir/daily/2015/05/20/input-data-files). Otherwise, the input paths " +
          s"are assumed to be flat directories of input files (e.g., trainDir/input-data-files). " +
          s"Default: ${defaultParams.trainDateRangeDaysAgoOpt}.")
        .action((x, c) => c.copy(trainDateRangeDaysAgoOpt = Some(x)))
      opt[String]("validate-input-dirs")
        .text("Input directories of validating data. Multiple input directories are also accepted if they are " +
          "separated by commas, e.g., inputDir1,inputDir2,inputDir3.")
        .action((x, c) => c.copy(validateDirsOpt = Some(x.split(","))))
      opt[String]("validate-date-range")
        .text(s"Date range for the validating data represented in the form start.date-end.date, " +
          s"e.g. 20150501-20150631. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., validateDir/daily/2015/05/20/input-data-files). Otherwise, the input " +
          s"paths are assumed to be flat directories of input files (e.g., validateDir/input-data-files). " +
          s"Default: ${defaultParams.validateDateRangeOpt}.")
        .action((x, c) => c.copy(validateDateRangeOpt = Some(x)))
      opt[String]("validate-date-range-days-ago")
        .text(s"Date range for the validating data represented in the form start.daysAgo-end.daysAgo, " +
          s"e.g. 90-1. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., validateDir/daily/2015/05/20/input-data-files). Otherwise, the input " +
          s"paths are assumed to be flat directories of input files (e.g., validateDir/input-data-files). " +
          s"Default: ${defaultParams.validateDateRangeDaysAgoOpt}.")
        .action((x, c) => c.copy(validateDateRangeDaysAgoOpt = Some(x)))
      opt[Int]("min-partitions-for-validation")
        .text(s"Minimum number of partitions for validating data (if provided). " +
          s"Default: ${defaultParams.minPartitionsForValidation}")
        .action((x, c) => c.copy(minPartitionsForValidation = x))
      opt[String]("output-dir")
        .required()
        .text(s"Output directory for logs and learned models.")
        .action((x, c) => c.copy(outputDir = x.replace(',', '_').replace(':', '_')))
      opt[String]("feature-name-and-term-set-path")
        .required()
        .text(s"Input path to the features name-and-term lists.")
        .action((x, c) => c.copy(featureNameAndTermSetInputPath = x))
      opt[String]("feature-shard-id-to-feature-section-keys-map")
        .text(s"A map between the feature shard id and it's corresponding feature section keys, in the following " +
          s"format: shardId1:sectionKey1,sectionKey2|shardId2:sectionKey2,sectionKey3.")
        .action((x, c) => c.copy(featureShardIdToFeatureSectionKeysMap =
          x.split("\\|")
            .map { line => line.split(":") match {
              case Array(key, names) => (key, names.split(",").map(_.trim).toSet)
              case Array(key) => (key, Set[String]())
            }}
            .toMap
        ))
      opt[Int]("num-iterations")
        .text(s"Number of coordinate descent iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[String]("updating-sequence")
        .text(s"Updating order of the ordinates (separated by commas) in the coordinate descent algorithm.")
        .action((x, c) => c.copy(updatingSequence = x.split(",")))
      opt[String]("fixed-effect-optimization-configurations")
        .text("Optimization configurations for the fixed effect optimization problem, multiple configurations are " +
          "accepted and separated by semi-colon \";\".")
        .action((x, c) => c.copy(fixedEffectOptimizationConfigurations =
          x.split(";")
            .map(
              _.split("\\|")
                .map { line =>
                  val Array(key, value) = line.split(":").map(_.trim)
                  (key, GLMOptimizationConfiguration.parseAndBuildFromString(value))
                }
                .toMap)
        ))
      opt[String]("fixed-effect-data-configurations")
        .text("Configurations for each fixed effect data set.")
        .action((x, c) => c.copy(fixedEffectDataConfigurations =
          x.split("\\|")
            .map { line =>
                val Array(key, value) = line.split(":").map(_.trim)
                (key, FixedEffectDataConfiguration.parseAndBuildFromString(value))
              }.toMap
        ))
      opt[String]("random-effect-optimization-configurations")
        .text("Optimization configurations for each random effect optimization problem, multiple parameters are " +
          "accepted and separated by semi-colon \";\".")
        .action((x, c) => c.copy(randomEffectOptimizationConfigurations =
          x.split(";")
            .map(
              _.split("\\|")
                .map { line =>
                  val Array(key, value) = line.split(":").map(_.trim)
                  (key, GLMOptimizationConfiguration.parseAndBuildFromString(value))
                }
                .toMap)
        ))
      opt[String]("factored-random-effect-optimization-configurations")
        .text("Optimization configurations for each factored random effect optimization problem, multiple parameters " +
          "are accepted and separated by semi-colon \";\".")
        .action((x, c) => c.copy(factoredRandomEffectOptimizationConfigurations =
          x.split(";")
            .map(
              _.split("\\|")
                .map { line =>
                  val Array(key, s1, s2, s3) = line.split(":").map(_.trim)
                  val randomEffectOptConfig = GLMOptimizationConfiguration.parseAndBuildFromString(s1)
                  val latentFactorOptConfig = GLMOptimizationConfiguration.parseAndBuildFromString(s2)
                  val mfOptimizationOptConfig = MFOptimizationConfiguration.parseAndBuildFromString(s3)
                  (key, (randomEffectOptConfig, latentFactorOptConfig, mfOptimizationOptConfig))
                }
                .toMap)
        ))
      opt[String]("random-effect-data-configurations")
        .text("Configurations for all the random effect data sets.")
        .action((x, c) => c.copy(randomEffectDataConfigurations =
          x.split("\\|")
            .map { line =>
              val Array(key, value) = line.split(":").map(_.trim)
              (key, RandomEffectDataConfiguration.parseAndBuildFromString(value))
            }
            .toMap
        ))
      opt[Boolean]("save-models-to-hdfs")
        .text(s"DEPRECATED -- USE model-output-mode")
        .action((x, c) => c.copy(modelOutputMode = if (x) ALL else NONE))
      opt[String]("model-output-mode")
        .text(s"Output mode of trained models to HDFS (ALL, BEST, or NONE)." +
          s"Default: ${defaultParams.modelOutputMode}")
        .action((x, c) => c.copy(modelOutputMode = ModelOutputMode.withName(x.toUpperCase)))
      opt[Int]("num-output-files-for-random-effect-model")
        .text(s"Number of output files to write for each random effect model. Not setting this parameter or " +
          s"setting it to -1 means to use the default number of output files." +
          s"Default: ${defaultParams.numberOfOutputFilesForRandomEffectModel}")
        .action((x, c) => c.copy(numberOfOutputFilesForRandomEffectModel = x))
      opt[String]("application-name")
        .text(s"Name of this Spark application. Default: ${defaultParams.applicationName}.")
        .action((x, c) => c.copy(applicationName = x))
      help("help")
        .text("prints usage text.")
    }
    parser.parse(args, Params()) match {
      case Some(parsedParams) => parsedParams
      case None => throw new IllegalArgumentException(s"Parsing the command line arguments failed " +
          s"(${args.mkString(", ")}),\n ${parser.usage}")
    }
  }
}
