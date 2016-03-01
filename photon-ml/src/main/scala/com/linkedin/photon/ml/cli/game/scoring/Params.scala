package com.linkedin.photon.ml.cli.game.scoring

import scala.collection.{Map, Set}

import scopt.OptionParser

import com.linkedin.photon.ml.supervised.TaskType
import com.linkedin.photon.ml.supervised.TaskType._


/**
 * @author xazhang
 */
case class Params(
    inputDirs: Array[String] = Array(),
    dateRangeOpt: Option[String] = None,
    featureNameAndTermSetInputPath: String = "",
    featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]] = Map(),
    randomEffectIdSet: Set[String] = Set(),
    gameModelInputDir: String = "",
    outputDir: String = "",
    numFiles: Int = 1,
    taskType: TaskType = LOGISTIC_REGRESSION,
    applicationName: String = "Game-Scoring") {

  override def toString = {
    s"Input parameters:\n" +
        s"inputDirs: ${inputDirs.mkString(", ")}\n" +
        s"dateRangeOpt: $dateRangeOpt\n" +
        s"featureNameAndTermSetInputPath: $featureNameAndTermSetInputPath\n" +
        s"featureShardIdToFeatureSectionKeysMap:\n${featureShardIdToFeatureSectionKeysMap.mapValues(_.mkString(", "))
            .mkString("\n")}\n" +
        s"randomEffectIdSet: $randomEffectIdSet\n" +
        s"gameModelInputDir: $gameModelInputDir\n" +
        s"outputDir: $outputDir\n" +
        s"numFiles: $numFiles\n" +
        s"taskType: $taskType\n" +
        s"applicationName: $applicationName"
  }
}

object Params {

  def parseFromCommandLine(args: Array[String]): Params = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("GLMix-Scoring-Params") {
      opt[String]("input-data-dirs")
        .required()
        .text("Input directories of data used to compute the viewer-actor-activityType affinity score. " +
        "Multiple input directories are separated by commas.")
        .action((x, c) => c.copy(inputDirs = x.split(",")))
      opt[String]("date-range")
        .text("Date range for the input data represented in the form start.date-end.date, e.g. 20150501-20150631, " +
        s"default: ${defaultParams.dateRangeOpt}.")
        .action((x, c) => c.copy(dateRangeOpt = Some(x)))
      opt[String]("feature-name-and-term-set-path")
          .required()
          .text("Input path to the features name-and-term lists.")
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
      opt[String]("random-effect-id-set")
        .required()
        .text("Comma separated set of random effect ids of the random effect models.")
        .action((x, c) => c.copy(randomEffectIdSet = x.split(",").toSet))
      opt[String]("game-model-input-dir")
        .required()
        .text(s"Input path of the distributed model used to generate affinity score.")
        .action((x, c) => c.copy(gameModelInputDir = x))
      opt[String]("output-dir")
        .required()
        .text(s"Output directory for the scores and log messages.")
        .action((x, c) => c.copy(outputDir = x.replaceAll(",|:", "_")))
      opt[Int]("num-files")
        .text(s"Number of files to store the scoring results, default: ${defaultParams.numFiles}.")
        .action((x, c) => c.copy(numFiles = x))
      opt[String]("application-name")
        .text(s"Name of this Spark application, default: ${defaultParams.applicationName}.")
        .action((x, c) => c.copy(applicationName = x))
      opt[String]("task-type")
        .required()
        .text("Task type. Examples include binary_classification and linear_regression.")
        .action((x, c) => c.copy(taskType = TaskType.withName(x.toUpperCase)))
      help("help")
        .text("Prints usage text")
    }
    parser.parse(args, Params()) match {
      case Some(parsedParams) => parsedParams
      case None => throw new IllegalArgumentException(s"Parsing the command line arguments failed " +
          s"(${args.mkString(", ")}),\n ${parser.usage}")
    }
  }
}
