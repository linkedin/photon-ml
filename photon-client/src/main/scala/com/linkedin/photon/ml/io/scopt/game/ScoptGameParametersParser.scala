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
package com.linkedin.photon.ml.io.scopt.game

import scala.language.existentials

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import org.joda.time.DateTimeZone
import scopt.OptionParser

import com.linkedin.photon.ml.DataValidationType
import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.Types.FeatureShardId
import com.linkedin.photon.ml.cli.game.GameDriver
import com.linkedin.photon.ml.data.InputColumnsNames
import com.linkedin.photon.ml.evaluation.EvaluatorType
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.io.scopt.{ScoptParameter, ScoptParser, ScoptParserHelpers, ScoptParserReads}
import com.linkedin.photon.ml.util.{DateRange, DaysRange, PhotonLogger}

/**
 * Base trait for any [[ScoptParser]] for a class derived from the [[GameDriver]].
 */
trait ScoptGameParametersParser extends ScoptParser {

  import ScoptParserReads._

  /**
   * Create a list of [[ScoptParameter]] instances wrapped around the [[GameDriver]] parameters.
   *
   * @param driver The [[GameDriver]] instance to use (important for [[org.apache.spark.ml.param.Param]] ownership)
   * @return The [[GameDriver]] parameters wrapped by [[ScoptParameter]] instances, in a single [[Seq]]
   */
  protected def createScoptGameParams(driver: GameDriver): Seq[ScoptParameter[In, Out] forSome { type In; type Out }] =
    Seq(
      // Input Data Directories
      ScoptParameter[Seq[Path], Set[Path]](
        driver.inputDataDirectories,
        parse = ScoptParserHelpers.parseSetFromSeq,
        print = ScoptParserHelpers.iterableToString,
        usageText = "<path1>,<path2>,...",
        isRequired = true),

      // Input Data Date Range
      ScoptParameter[DateRange, DateRange](
        driver.inputDataDateRange,
        usageText = s"${DateRange.DEFAULT_PATTERN}${DateRange.DEFAULT_DELIMITER}${DateRange.DEFAULT_PATTERN}"),

      // Input Data Days Range
      ScoptParameter[DaysRange, DaysRange](
        driver.inputDataDaysRange,
        usageText = s"xx${DaysRange.DEFAULT_DELIMITER}xx"),

      // Off-heap Index Map Directory
      ScoptParameter[Path, Path](
        driver.offHeapIndexMapDirectory,
        usageText = "<path>"),

      // Off-heap Index Map Partitions
      ScoptParameter[Int, Int](
        driver.offHeapIndexMapPartitions,
        usageText = "<value>"),

      // Feature Bags Directory
      ScoptParameter[Path, Path](
        driver.featureBagsDirectory,
        usageText = "<path>"),

      // Input Column Names
      ScoptParameter[Map[String, String], InputColumnsNames](
        driver.inputColumnNames,
        parse = ScoptParserHelpers.parseInputColumnNames,
        print = ScoptParserHelpers.inputColumnNamesToString,
        usageText = "<col>=<name>,<col>=<name>,...",
        additionalDocs = Seq(s"base column names: ${InputColumnsNames.all.map(_.toString).mkString(", ")}")),

      // Evaluators
      ScoptParameter[Seq[EvaluatorType], Seq[EvaluatorType]](
        driver.evaluators,
        print = ScoptParserHelpers.iterableToString,
        usageText = "<eval1>,<eval2>,...",
        additionalDocs = Seq(s"example evaluator types: ${EvaluatorType.all.map(_.name).mkString(", ")}, etc.")),

      // Root Output Directory
      ScoptParameter[Path, Path](
        driver.rootOutputDirectory,
        usageText = "<path>",
        isRequired = true),

      // Override Output Directory
      ScoptParameter[Boolean, Boolean](
        driver.overrideOutputDirectory),

      // Output Files Limit
      ScoptParameter[Int, Int](
        driver.outputFilesLimit,
        usageText = "<value>"),

      // Feature Shard Configurations
      ScoptParameter[Map[String, String], Map[FeatureShardId, FeatureShardConfiguration]](
        driver.featureShardConfigurations,
        parse = ScoptParserHelpers.parseFeatureShardConfiguration,
        updateOpt = Some(ScoptParserHelpers.updateFeatureShardConfigurations),
        printSeq = ScoptParserHelpers.featureShardConfigsToStrings,
        usageText = "<arg>=<value>",
        additionalDocs = Seq(
          s"required args: ${ScoptParserHelpers.formatArgs(ScoptParserHelpers.FEATURE_SHARD_CONFIG_REQUIRED_ARGS)}",
          s"optional args: ${ScoptParserHelpers.formatArgs(ScoptParserHelpers.FEATURE_SHARD_CONFIG_OPTIONAL_ARGS)}"),
        isRequired = true),

      // Data Validation
      ScoptParameter[DataValidationType, DataValidationType](
        driver.dataValidation,
        usageText = "<type>",
        additionalDocs = Seq(s"data validation types: ${DataValidationType.values.map(_.toString).mkString(", ")}")),

      // Log Level
      ScoptParameter[String, Int](
        driver.logLevel,
        parse = PhotonLogger.parseLogLevelString,
        print = PhotonLogger.printLogLevelString,
        usageText = "<level>",
        additionalDocs = Seq(s"log levels: ${PhotonLogger.logLevelNames.keys.map(_.toString).mkString(", ")}")),

      // Application Name
      ScoptParameter[String, String](
        driver.applicationName,
        usageText = "<name>"),

      // Time zone
      ScoptParameter[DateTimeZone, DateTimeZone](
        driver.timeZone,
        usageText = "<time zone>",
        additionalDocs = Seq("For a list of valid timezone ids, see: http://joda-time.sourceforge.net/timezones.html")))

  // Parser for reading command line input into a ParamMap
  protected def parser: OptionParser[ParamMap]

  /**
   * Parse command line arguments for GAME training into a [[org.apache.spark.ml.param.ParamMap]].
   *
   * @param args [[Array]] of command line arguments
   * @return An initialized [[ParamMap]]
   */
  def parseFromCommandLine(args: Array[String]): ParamMap =
    parser.parse(args, ParamMap.empty) match {
      case Some(params) => params

      case None =>
        val errMsg = args
          .grouped(2)
          .map(_.mkString(" "))
          .mkString("\n")

        throw new IllegalArgumentException(s"Parsing the following command line arguments failed:\n${errMsg.toString}")
    }

  /**
   * Convert parameters stored in a valid [[ParamMap]] object to [[String]] format for output to the command line, in a
   * format which can be parsed back into a valid [[ParamMap]].
   *
   * @param paramMap A valid [[ParamMap]]
   * @return A [[Seq]] of [[String]] representations of the parameters, in a format that can be parsed by Scopt
   */
  def printForCommandLine(paramMap: ParamMap): Seq[String]
}
