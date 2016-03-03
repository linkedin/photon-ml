package com.linkedin.photon.ml.cli.game.training

import scala.collection.{Map, Set}

import scopt.OptionParser

import com.linkedin.photon.ml.optimization.game.{MFOptimizationConfiguration, GLMOptimizationConfiguration}
import com.linkedin.photon.ml.data.{FixedEffectDataConfiguration, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.supervised.TaskType
import com.linkedin.photon.ml.supervised.TaskType._


/**
 * @author xazhang
 */
case class Params(
    trainDirs: Array[String] = Array(),
    trainDateRangeOpt: Option[String] = None,
    numDaysDataForTraining: Option[Int] = None,
    validateDirsOpt: Option[Array[String]] = None,
    validateDateRangeOpt: Option[String] = None,
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
    isSavingModelsToHDFS: Boolean = true,
    applicationName: String = "Game-Full-Model-Training") {

  override def toString = {
    s"trainDirs: ${trainDirs.mkString(", ")}\n" +
        s"trainDateRangeOpt: $trainDateRangeOpt\n" +
        s"numDaysDataForTraining: $numDaysDataForTraining\n" +
        s"validateDirsOpt: ${validateDirsOpt.map(_.mkString(", "))}\n" +
        s"validateDateRangeOpt: $validateDateRangeOpt\n" +
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
        s"saveModelsToHDFS: $isSavingModelsToHDFS\n" +
        s"applicationName: $applicationName"
  }
}

object Params {
  def parseFromCommandLine(args: Array[String]): Params = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Photon-Game") {
      opt[String]("train-input-dirs")
        .required()
        .text("input directories of training data in response prediction AVRO format. " +
          "Multiple input directories are separated by commas.")
        .action((x, c) => c.copy(trainDirs = x.split(",")))
      opt[String]("task-type")
        .required()
        .text("Task type. Examples include logistic_regression and linear_regression.")
        .action((x, c) => c.copy(taskType = TaskType.withName(x.toUpperCase)))
      opt[String]("train-date-range")
        .text(s"Date range for the training data represented in the form start.date-end.date, " +
          s"e.g. 20150501-20150631, default: ${defaultParams.trainDateRangeOpt}.")
        .action((x, c) => c.copy(trainDateRangeOpt = Some(x)))
      opt[Int]("num-days-data-for-training")
        .text(s"Number of days of data used for training. Currently this parameter is only used in the daily " +
          s"training pipeline. Default: ${defaultParams.numDaysDataForTraining}.")
        .action((x, c) => c.copy(numDaysDataForTraining = Some(x)))
      opt[String]("validate-input-dirs")
        .text(s"Input directories of validating data in response prediction AVRO format, " +
          s"multiple input directories are separated by commas." +
        s"Default: ${defaultParams.validateDirsOpt}")
        .action((x, c) => c.copy(validateDirsOpt = Some(x.split(","))))
      opt[String]("validate-date-range")
        .text(s"date range for the validating data represented in the form start.date-end.date," +
          s" e.g. 20150501-20150631, default: ${defaultParams.validateDateRangeOpt}.")
        .action((x, c) => c.copy(validateDateRangeOpt = Some(x)))
      opt[Int]("min-partitions-for-validation")
        .text(s"Minimum number of partitions for validating data. Default: ${defaultParams.minPartitionsForValidation}")
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
          x.split("\\|").map { line => line.split(":") match {
            case Array(key, names) => (key, names.split(",").map(_.trim).toSet)
            case Array(key) => (key, Set[String]())
          }
          }.toMap
      ))
      opt[Int]("num-iterations")
        .text(s"Number of outer iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[String]("updating-sequence")
        .text(s"Updating sequence of the ordinates (separated by commas) in the coordinate descent.")
        .action((x, c) => c.copy(updatingSequence = x.split(",")))
      opt[String]("fixed-effect-optimization-configurations")
        .text("Optimization configurations for the fixed effect optimization problem, multiple configurations are " +
          "separated by semi-colon \";\".")
        .action((x, c) => c.copy(fixedEffectOptimizationConfigurations =
          x.split(";").map(_.split("\\|").map { line => val Array(key, value) = line.split(":").map(_.trim)
            (key, GLMOptimizationConfiguration.parseAndBuildFromString(value))
          }.toMap)
      ))
      opt[String]("fixed-effect-data-configurations")
        .text("Configurations for each fixed effect data set.")
        .action((x, c) => c.copy(fixedEffectDataConfigurations =
          x.split("\\|").map { line => val Array(key, value) = line.split(":").map(_.trim)
            (key, FixedEffectDataConfiguration.parseAndBuildFromString(value))
          }.toMap
      ))
      opt[String]("random-effect-optimization-configurations")
        .text("Optimization configurations for each random effect optimization problem, multiple parameters are " +
          "separated by semi-colon \";\".")
        .action((x, c) => c.copy(randomEffectOptimizationConfigurations =
        x.split(";").map(_.split("\\|").map { line => val Array(key, value) = line.split(":").map(_.trim)
          (key, GLMOptimizationConfiguration.parseAndBuildFromString(value))
        }.toMap)
      ))
      opt[String]("factored-random-effect-optimization-configurations")
          .text("Optimization configurations for each random effect optimization problem, multiple parameters are " +
          "separated by semi-colon \";\".")
          .action((x, c) => c.copy(factoredRandomEffectOptimizationConfigurations =
          x.split(";").map(_.split("\\|").map { line => val Array(key, s1, s2, s3) = line.split(":").map(_.trim)
            val randomEffectOptConfig = GLMOptimizationConfiguration.parseAndBuildFromString(s1)
            val latentFactorOptConfig = GLMOptimizationConfiguration.parseAndBuildFromString(s2)
            val mfOptimizationOptConfig = MFOptimizationConfiguration.parseAndBuildFromString(s3)
            (key, (randomEffectOptConfig, latentFactorOptConfig, mfOptimizationOptConfig))
          }.toMap)
      ))
      opt[String]("random-effect-data-configurations")
        .text("Configurations for each random effect data set.")
        .action((x, c) => c.copy(randomEffectDataConfigurations =
        x.split("\\|").map { line => val Array(key, value) = line.split(":").map(_.trim)
          (key, RandomEffectDataConfiguration.parseAndBuildFromString(value))
        }.toMap
      ))
      opt[Boolean]("save-models-to-hdfs")
        .text(s"Whether to save the models (best model and all models) to HDFS. " +
          s"Default: ${defaultParams.isSavingModelsToHDFS}")
        .action((x, c) => c.copy(isSavingModelsToHDFS = x))
      opt[String]("application-name")
        .text(s"name of this Spark application, default: ${defaultParams.applicationName}.")
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
