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
package com.linkedin.photon.ml.cli.game.scoring

import scala.collection.{Map, Set}

import scopt.OptionParser

/**
 * Command line arguments for GAME scoring driver
 *
 * @param inputDirs Input directories of data to be scored. Multiple input directories are also accepted if they are
 *                  separated by commas, e.g., inputDir1,inputDir2,inputDir3.
 * @param dateRangeOpt Date range for the input data represented in the form start.date-end.date,
 *                     e.g. 20150501-20150631. If dateRangeOpt is specified, the input directory is expected
 *                     to be in the daily format structure (e.g., inputDir/daily/2015/05/20/input-data-files).
 *                     Otherwise, the input paths are assumed to be flat directories of input files
 *                     (e.g., inputDir/input-data-files).
 * @param dateRangeDaysAgoOpt Date range for the input data represented in the form start.daysAgo-end.daysAgo,
 *                            e.g. 90-1. If dateRangeDaysAgoOpt is specified, the input directory is expected
 *                            to be in the daily format structure (e.g., inputDir/daily/2015/05/20/input-data-files).
 *                            Otherwise, the input paths are assumed to be flat directories of input files
 *                            (e.g., inputDir/input-data-files).
 * @param featureNameAndTermSetInputPath Input path to the features name-and-term lists.
 * @param featureShardIdToFeatureSectionKeysMap A map between the feature shard id and it's corresponding feature
 *                                              section keys in the following format:
 *                                              shardId1:sectionKey1,sectionKey2|shardId2:sectionKey2,sectionKey3.
 * @param randomEffectIdSet A set of random effect ids of the corresponding random effect models in the following
 *                          format: randomEffectId1,randomEffectId2,randomEffectId3,
 * @param minPartitionsForRandomEffectModel Minimum number of partitions for GAME's random effect model
 * @param gameModelInputDir Input directory of the GAME model to be used to for scoring purpose
 * @param outputDir Output directory for logs and the scores.
 * @param numOutputFilesForScores Number of output files to write for the computed scores.
 * @param applicationName Name of this Spark application.
 */
case class Params(
    inputDirs: Array[String] = Array(),
    dateRangeOpt: Option[String] = None,
    dateRangeDaysAgoOpt: Option[String] = None,
    featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]] = Map(),
    featureShardIdToInterceptMap: Map[String, Boolean] = Map(),
    randomEffectIdSet: Set[String] = Set(),
    minPartitionsForRandomEffectModel: Int = 1,
    featureNameAndTermSetInputPath: String = "",
    gameModelInputDir: String = "",
    outputDir: String = "",
    numOutputFilesForScores: Int = -1,
    applicationName: String = "Game-Scoring") {

  override def toString: String = {
    s"inputDirs: ${inputDirs.mkString(", ")}\n" +
      s"dateRangeOpt: $dateRangeOpt\n" +
      s"dateRangeDaysAgoOpt: $dateRangeDaysAgoOpt\n" +
      s"featureShardIdToFeatureSectionKeysMap:\n${featureShardIdToFeatureSectionKeysMap.mapValues(_.mkString(", "))
            .mkString("\n")}\n" +
      s"featureShardIdToInterceptMap:\n${featureShardIdToInterceptMap.mkString("\n")}" +
      s"featureNameAndTermSetInputPath: $featureNameAndTermSetInputPath\n" +
      s"randomEffectIdSet: $randomEffectIdSet\n" +
      s"numPartitionsForRandomEffectModel: $minPartitionsForRandomEffectModel\n" +
      s"gameModelInputDir: $gameModelInputDir\n" +
      s"outputDir: $outputDir\n" +
      s"numOutputFilesForScores: $numOutputFilesForScores\n" +
      s"applicationName: $applicationName"
  }
}

object Params {

  def parseFromCommandLine(args: Array[String]): Params = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("GLMix-Scoring-Params") {
      opt[String]("input-data-dirs")
        .required()
        .text("Input directories of data to be scored. Multiple input directories are also accepted if they are " +
          "separated by commas, e.g., inputDir1,inputDir2,inputDir3.")
        .action((x, c) => c.copy(inputDirs = x.split(",")))
      opt[String]("date-range")
          .text(s"Date range for the input data represented in the form start.date-end.date, " +
              s"e.g. 20150501-20150631. If this parameter is specified, the input directory is expected to be in the " +
              s"daily format structure (e.g., inputDir/daily/2015/05/20/input-data-files). Otherwise, the input paths " +
              s"are assumed to be flat directories of input files (e.g., inputDir/input-data-files). " +
              s"Default: ${defaultParams.dateRangeOpt}.")
          .action((x, c) => c.copy(dateRangeOpt = Some(x)))
      opt[String]("date-range-days-ago")
          .text(s"Date range for the input data represented in the form start.daysAgo-end.daysAgo, " +
              s"e.g. 90-1. If this parameter is specified, the input directory is expected to be in the " +
              s"daily format structure (e.g., inputDir/daily/2015/05/20/input-data-files). Otherwise, the input paths " +
              s"are assumed to be flat directories of input files (e.g., inputDir/input-data-files). " +
              s"Default: ${defaultParams.dateRangeDaysAgoOpt}.")
          .action((x, c) => c.copy(dateRangeDaysAgoOpt = Some(x)))
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
              }}.toMap
          ))
      opt[String]("feature-shard-id-to-intercept-map")
          .text(s"A map between the feature shard id and a boolean variable to decide whether a dummy feature " +
              s"should be added to this shard to learn an intercept. The default is true for all shard ids.")
          .action((x, c) => c.copy(featureShardIdToInterceptMap =
              x.split("\\|").map { line => line.split(":") match {
                case Array(key, flag) => (key, flag.toBoolean)
                case Array(key) => (key, true)
              }}.toMap
          ))
      opt[String]("random-effect-id-set")
        .required()
        .text("A set of random effect ids of the corresponding random effect models in the following format: " +
          s"randomEffectId1,randomEffectId2,randomEffectId3, Default: ${defaultParams.randomEffectIdSet}")
        .action((x, c) => c.copy(randomEffectIdSet = x.split(",").toSet))
      opt[String]("game-model-input-dir")
        .required()
        .text(s"Input directory of the GAME model to be used to for scoring purpose.")
        .action((x, c) => c.copy(gameModelInputDir = x))
      opt[String]("output-dir")
        .required()
        .text(s"Output directory for logs and the scores.")
        .action((x, c) => c.copy(outputDir = x.replaceAll(",|:", "_")))
      opt[Int]("num-files")
        .text("Number of output files to write for the computed scores. " +
          s"Default: ${defaultParams.numOutputFilesForScores}")
        .action((x, c) => c.copy(numOutputFilesForScores = x))
      opt[String]("application-name")
        .text(s"Name of this Spark application. Default: ${defaultParams.applicationName}")
        .action((x, c) => c.copy(applicationName = x))
      //TODO: remove the task-type option
      opt[String]("task-type")
        .text("A dummy option that does nothing and will be removed for the next major version bump")
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
