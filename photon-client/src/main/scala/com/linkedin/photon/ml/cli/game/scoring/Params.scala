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
package com.linkedin.photon.ml.cli.game.scoring

import scopt.OptionParser

import com.linkedin.photon.ml.PhotonOptionNames._
import com.linkedin.photon.ml.cli.game.{EvaluatorParams, FeatureParams}
import com.linkedin.photon.ml.util.{Utils, PalDBIndexMapParams}

/**
 * Command line arguments for GAME scoring driver.
 */
class Params extends FeatureParams with PalDBIndexMapParams with EvaluatorParams {

  /**
   * Input directories of data to be scored. Multiple input directories are also accepted if they are separated by
   * commas, e.g., inputDir1,inputDir2,inputDir3.
   */
  var inputDirs: Array[String] = Array()

  /**
   * Date range for the input data represented in the form start.date-end.date, e.g. 20150501-20150631. If dateRangeOpt
   * is specified, the input directory is expected to be in the daily format structure
   * (e.g., inputDir/daily/2015/05/20/input-data-files). Otherwise, the input paths are assumed to be flat directories
   * of input files (e.g., inputDir/input-data-files).
   */
  var dateRangeOpt: Option[String] = None

  /**
   * Date range for the input data represented in the form start.daysAgo-end.daysAgo, e.g. 90-1. If dateRangeDaysAgoOpt
   * is specified, the input directory is expected to be in the daily format structure
   * (e.g., inputDir/daily/2015/05/20/input-data-files). Otherwise, the input paths are assumed to be flat directories
   * of input files (e.g., inputDir/input-data-files).
   */
  var dateRangeDaysAgoOpt: Option[String] = None

  /**
   * A set of random effect types of the corresponding random effect models in the following format:
   * randomEffectType1,randomEffectType2,randomEffectType3
   */
  var randomEffectTypeSet: Set[String] = Set()

  /**
   * Minimum number of partitions for GAME's random effect model
   */
  var minPartitionsForRandomEffectModel: Int = 1

  /**
   * Input directory of the GAME model to be used to for scoring purpose
   */
  var gameModelInputDir: String = ""

  /**
   * The GAME model's id that is used to populate the "modelId" field of ScoringResultAvro (output format of the
   * computed scores).
   */
  var gameModelId: String = ""

  /**
    * Flag to decide whether the data and model statistics should be logged. This adds an additional linear scan of the
    * dataset which can be time-consuming for very large datasets
    */
  var logDatasetAndModelStats: Boolean = false

  /**
   * Output directory for logs in text file and the scores in ScoringResultAvro format.
   */
  var outputDir: String = ""

  /**
   * Number of output files to write for the computed scores.
   */
  var numOutputFilesForScores: Int = -1

  /**
   * Whether to delete the output directory if exists
   */
  var deleteOutputDirIfExists: Boolean = false

  /**
   * Name of this Spark application.
   */
  var applicationName: String = "Game-Scoring"

  /**
   *
   * @return
   */
  override def toString: String = {
    s"inputDirs: ${inputDirs.mkString(", ")}\n" +
      s"dateRangeOpt: $dateRangeOpt\n" +
      s"dateRangeDaysAgoOpt: $dateRangeDaysAgoOpt\n" +
      s"featureShardIdToFeatureSectionKeysMap:\n${featureShardIdToFeatureSectionKeysMap.mapValues(_.mkString(", "))
            .mkString("\n")}\n" +
      s"featureShardIdToInterceptMap:\n${featureShardIdToInterceptMap.mkString("\n")}" +
      s"featureNameAndTermSetInputPath: $featureNameAndTermSetInputPath\n" +
      s"randomEffectTypeSet: $randomEffectTypeSet\n" +
      s"numPartitionsForRandomEffectModel: $minPartitionsForRandomEffectModel\n" +
      s"gameModelInputDir: $gameModelInputDir\n" +
      s"gameModelId: $gameModelId\n" +
      s"logDatasetAndModelStats: $logDatasetAndModelStats\n" +
      s"outputDir: $outputDir\n" +
      s"numOutputFilesForScores: $numOutputFilesForScores\n" +
      s"deleteOutputDirIfExists: $deleteOutputDirIfExists\n" +
      s"evaluatorTypes: ${evaluatorTypes.map(_.name).mkString("\t")}\n" +
      s"applicationName: $applicationName\n" +
      s"offHeapIndexMapDir: $offHeapIndexMapDir\n" +
      s"offHeapIndexMapNumPartitions: $offHeapIndexMapNumPartitions"
  }
}

object Params {

  /**
   *
   * @param args
   * @return
   */
  def parseFromCommandLine(args: Array[String]): Params = {
    val defaultParams = new Params()
    val params = new Params()

    val parser = new OptionParser[Unit]("GLMix-Scoring-Params") {
      opt[String]("input-data-dirs")
        .required()
        .text("Input directories of data to be scored. Multiple input directories are also accepted if they are " +
          "separated by commas, e.g., inputDir1,inputDir2,inputDir3.")
        .foreach(x => params.inputDirs = x.split(","))

      opt[String]("date-range")
        .text(s"Date range for the input data represented in the form start.date-end.date, " +
          s"e.g. 20150501-20150631. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., inputDir/daily/2015/05/20/input-data-files). Otherwise, the input paths " +
          s"are assumed to be flat directories of input files (e.g., inputDir/input-data-files). " +
          s"Default: ${defaultParams.dateRangeOpt}.")
        .foreach(x => params.dateRangeOpt = Some(x))

      opt[String]("date-range-days-ago")
        .text(s"Date range for the input data represented in the form start.daysAgo-end.daysAgo, " +
          s"e.g. 90-1. If this parameter is specified, the input directory is expected to be in the " +
          s"daily format structure (e.g., inputDir/daily/2015/05/20/input-data-files). Otherwise, the input paths " +
          s"are assumed to be flat directories of input files (e.g., inputDir/input-data-files). " +
          s"Default: ${defaultParams.dateRangeDaysAgoOpt}.")
        .foreach(x => params.dateRangeDaysAgoOpt = Some(x))

      opt[String]("feature-name-and-term-set-path")
        .required()
        .text("Input path to the features name-and-term lists.\n" +
          s"DEPRECATED -- This option will be removed in the next major version. Use the offheap index map " +
          s"configuration instead")
        .foreach(x => params.featureNameAndTermSetInputPath = x)

      opt[String]("feature-shard-id-to-feature-section-keys-map")
        .text(s"A map between the feature shard id and it's corresponding feature section keys, in the following " +
          s"format: shardId1:sectionKey1,sectionKey2|shardId2:sectionKey2,sectionKey3.")
        .foreach(x => params.featureShardIdToFeatureSectionKeysMap =
            x.split("\\|")
              .map { line => line.split(":") match {
                case Array(key, names) => (key, names.split(",").map(_.trim).toSet)
                case Array(key) => (key, Set[String]())
              }}
              .toMap)

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
              .toMap)

      opt[String]("random-effect-id-set")
        .text("A set of random effect types of the corresponding random effect models in the following format: " +
          s"randomEffectType1,randomEffectType2,randomEffectType3, Default: ${defaultParams.randomEffectTypeSet}")
        .foreach(x => params.randomEffectTypeSet = x.split(",").toSet)

      opt[String]("game-model-id")
        .text(s"The GAME model's id that is used to populate the 'modelId' field of ScoringResultAvro " +
          s"(output format of the computed scores). Default: ${defaultParams.gameModelId}")
        .foreach(x => params.gameModelId = x)

      opt[String]("game-model-input-dir")
        .required()
        .text(s"Input directory of the GAME model to be used to for scoring purpose.")
        .foreach(x => params.gameModelInputDir = x)

      opt[Boolean]("log-game-dataset-and-model-stats")
        .text(s"Whether to log stats about the dataset and models. Default: ${defaultParams.logDatasetAndModelStats}")
        .foreach(x => params.logDatasetAndModelStats = x)

      opt[String]("output-dir")
        .required()
        .text(s"Output directory for logs and the scores.")
        .foreach(x => params.outputDir = x.replaceAll(",|:", "_"))

      opt[Int]("num-files")
        .text("Number of output files to write for the computed scores. " +
          s"Default: ${defaultParams.numOutputFilesForScores}")
        .foreach(x => params.numOutputFilesForScores = x)

      opt[String]("application-name")
        .text(s"Name of this Spark application. Default: ${defaultParams.applicationName}")
        .foreach(x => params.applicationName = x)

      opt[Boolean]("delete-output-dir-if-exists")
        .text(s"Whether to delete the output directory if exists. Default: ${defaultParams.deleteOutputDirIfExists}")
        .foreach(x => params.deleteOutputDirIfExists = x)

      opt[String]("evaluator-type")
        .text("Type of the evaluator used to evaluate the computed scores.")
        .foreach(x => params.evaluatorTypes = x.split(",").map(Utils.evaluatorWithName _))

      // TODO: Remove the task-type option
      opt[String]("task-type")
        .text("A dummy option that does nothing and will be removed for the next major version bump")

      opt[String](OFFHEAP_INDEXMAP_DIR)
        .text("The offheap storage directory if offheap map is needed. DefaultIndexMap will be used if not specified.")
        .foreach(x => params.offHeapIndexMapDir = Some(x))

      opt[Int](OFFHEAP_INDEXMAP_NUM_PARTITIONS)
        .text("The number of partitions for the offheap map storage. This partition number should be consistent with " +
            "the number when offheap storage is built. This parameter affects only the execution speed during " +
            "feature index building and has zero performance impact on training other than maintaining a " +
            "convention.")
        .foreach(x => params.offHeapIndexMapNumPartitions = x)

      help("help").text("Prints usage text")
    }

    if (!parser.parse(args)) {
      throw new IllegalArgumentException(s"Parsing the command line arguments failed.\n" +
        s"(${args.mkString(", ")}),\n ${parser.usage}")
    }

    params
  }
}
