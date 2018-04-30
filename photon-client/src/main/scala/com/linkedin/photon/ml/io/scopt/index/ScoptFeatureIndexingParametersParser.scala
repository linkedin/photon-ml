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
package com.linkedin.photon.ml.io.scopt.index

import scala.language.existentials

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import org.joda.time.DateTimeZone
import scopt.{OptionDef, OptionParser, Read}

import com.linkedin.photon.ml.Types.FeatureShardId
import com.linkedin.photon.ml.index.FeatureIndexingDriver
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.io.scopt.{ScoptParameter, ScoptParser, ScoptParserHelpers, ScoptParserReads}
import com.linkedin.photon.ml.util.{DateRange, DaysRange}

/**
 * Scopt command line argument parser for feature indexing job parameters.
 */
object ScoptFeatureIndexingParametersParser extends ScoptParser {

  import ScoptParserReads._

  val scoptFeatureIndexingParams: Seq[ScoptParameter[In, Out] forSome { type In; type Out }] = Seq(

    // Input Data Directories
    ScoptParameter[Seq[Path], Set[Path]](
      FeatureIndexingDriver.inputDataDirectories,
      parse = ScoptParserHelpers.parseSetFromSeq,
      print = ScoptParserHelpers.iterableToString,
      usageText = "<path1>,<path2>,...",
      isRequired = true),

    // Input Data Date Range
    ScoptParameter[DateRange, DateRange](
      FeatureIndexingDriver.inputDataDateRange,
      usageText = s"${DateRange.DEFAULT_PATTERN}${DateRange.DEFAULT_DELIMITER}${DateRange.DEFAULT_PATTERN}"),

    // Input Data Days Range
    ScoptParameter[DaysRange, DaysRange](
      FeatureIndexingDriver.inputDataDaysRange,
      usageText = s"xx${DaysRange.DEFAULT_DELIMITER}xx"),

    // Minimum Input Partitions
    ScoptParameter[Int, Int](
      FeatureIndexingDriver.minInputPartitions,
      usageText = "<value>"),

    // Root Output Directory
    ScoptParameter[Path, Path](
      FeatureIndexingDriver.rootOutputDirectory,
      usageText = "<path>",
      isRequired = true),

    // Override Output Directory
    ScoptParameter[Boolean, Boolean](
      FeatureIndexingDriver.overrideOutputDirectory),

    // Num Partitions
    ScoptParameter[Int, Int](
      FeatureIndexingDriver.numPartitions,
      usageText = "<value>",
      isRequired = true),

    // Feature Shard Configurations
    ScoptParameter[Map[String, String], Map[FeatureShardId, FeatureShardConfiguration]](
      FeatureIndexingDriver.featureShardConfigurations,
      parse = ScoptParserHelpers.parseFeatureShardConfiguration,
      updateOpt = Some(ScoptParserHelpers.updateFeatureShardConfigurations),
      printSeq = ScoptParserHelpers.featureShardConfigsToStrings,
      usageText = "<arg>=<value>",
      additionalDocs = Seq(
        s"required args: ${ScoptParserHelpers.formatArgs(ScoptParserHelpers.FEATURE_SHARD_CONFIG_REQUIRED_ARGS)}",
        s"optional args: ${ScoptParserHelpers.formatArgs(ScoptParserHelpers.FEATURE_SHARD_CONFIG_OPTIONAL_ARGS)}"),
      isRequired = true),

    // Application Name
    ScoptParameter[String, String](
      FeatureIndexingDriver.applicationName,
      usageText = "<name>"),

    // Time zone
    ScoptParameter[DateTimeZone, DateTimeZone](
      FeatureIndexingDriver.timeZone,
      usageText = "<time zone>",
      additionalDocs = Seq("For a list of valid timezone ids, see: http://joda-time.sourceforge.net/timezones.html")))

  /**
   * Parse command line arguments for feature indexing into a [[ParamMap]].
   *
   * @param args [[Array]] of command line arguments
   * @return An initialized [[ParamMap]]
   */
  def parseFromCommandLine(args: Array[String]): ParamMap = {

    val parser = new OptionParser[ParamMap]("Feature-Indexing") {

      private def optHelper[In](scoptParameter: ScoptParameter[In, _]): OptionDef[In, ParamMap] = {

        implicit val read: Read[In] = scoptParameter.read

        scoptParameter.toOptionDef(opt[In])
      }

      scoptFeatureIndexingParams.foreach { optHelper(_) }
    }

    parser.parse(args, ParamMap.empty) match {
      case Some(params) => params

      case None =>
        val errMsg = args
          .grouped(2)
          .map(_.mkString(" "))
          .mkString("\n")

        throw new IllegalArgumentException(s"Parsing the following command line arguments failed:\n${errMsg.toString()}")
    }
  }

  /**
   * Given a [[ParamMap]] of valid parameters, convert them into a [[Seq]] of [[String]] representations which can be
   * parsed by Scopt.
   *
   * @param paramMap Valid feature indexing parameters
   * @return A [[Seq]] of [[String]] representations of the parameters, in a format that can be parsed by Scopt
   */
  def printForCommandLine(paramMap: ParamMap): Seq[String] = {

    FeatureIndexingDriver.validateParams(paramMap)

    scoptFeatureIndexingParams.flatMap(_.generateCmdLineArgs(paramMap))
  }
}
