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
package com.linkedin.photon.ml.io.scopt.avro

import scala.language.existentials

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import org.joda.time.DateTimeZone
import scopt.{OptionDef, OptionParser, Read}

import com.linkedin.photon.ml.data.avro.NameAndTermFeatureBagsDriver
import com.linkedin.photon.ml.io.scopt.{ScoptParameter, ScoptParser, ScoptParserHelpers, ScoptParserReads}
import com.linkedin.photon.ml.util.{DateRange, DaysRange}

/**
 * Scopt command line argument parser for name-and-term feature bag generation job parameters.
 */
object ScoptNameAndTermFeatureBagsParametersParser extends ScoptParser {

  import ScoptParserReads._

  val scoptNameAndTermFeatureBagsParams: Seq[ScoptParameter[In, Out] forSome { type In; type Out }] = Seq(

    // Input Data Directories
    ScoptParameter[Seq[Path], Set[Path]](
      NameAndTermFeatureBagsDriver.inputDataDirectories,
      parse = ScoptParserHelpers.parseSetFromSeq,
      print = ScoptParserHelpers.iterableToString,
      usageText = "<path1>,<path2>,...",
      isRequired = true),

    // Input Data Date Range
    ScoptParameter[DateRange, DateRange](
      NameAndTermFeatureBagsDriver.inputDataDateRange,
      usageText = s"${DateRange.DEFAULT_PATTERN}${DateRange.DEFAULT_DELIMITER}${DateRange.DEFAULT_PATTERN}"),

    // Input Data Days Range
    ScoptParameter[DaysRange, DaysRange](
      NameAndTermFeatureBagsDriver.inputDataDaysRange,
      usageText = s"xx${DaysRange.DEFAULT_DELIMITER}xx"),

    // Minimum Input Partitions
    ScoptParameter[Int, Int](
      NameAndTermFeatureBagsDriver.minInputPartitions,
      usageText = "<value>"),

    // Root Output Directory
    ScoptParameter[Path, Path](
      NameAndTermFeatureBagsDriver.rootOutputDirectory,
      usageText = "<path>",
      isRequired = true),

    // Override Output Directory
    ScoptParameter[Boolean, Boolean](
      NameAndTermFeatureBagsDriver.overrideOutputDirectory),

    // Feature Bags Keys
    ScoptParameter[Seq[String], Set[String]](
      NameAndTermFeatureBagsDriver.featureBagsKeys,
      parse = ScoptParserHelpers.parseSetFromSeq,
      print = ScoptParserHelpers.iterableToString,
      usageText = "<bag1>,<bag2>,...",
      isRequired = true),

    // Application Name
    ScoptParameter[String, String](
      NameAndTermFeatureBagsDriver.applicationName,
      usageText = "<name>"),

    // Time zone
    ScoptParameter[DateTimeZone, DateTimeZone](
      NameAndTermFeatureBagsDriver.timeZone,
      usageText = "<time zone>",
      additionalDocs = Seq("For a list of valid timezone ids, see: http://joda-time.sourceforge.net/timezones.html")))

  /**
   * Parse command line arguments for name-and-term generation into a [[ParamMap]].
   *
   * @param args [[Array]] of command line arguments
   * @return An initialized [[ParamMap]]
   */
  def parseFromCommandLine(args: Array[String]): ParamMap = {

    val parser = new OptionParser[ParamMap]("Name-And-Term-Feature-Bag-Generation") {

      private def optHelper[In](scoptParameter: ScoptParameter[In, _]): OptionDef[In, ParamMap] = {

        implicit val read: Read[In] = scoptParameter.read

        scoptParameter.toOptionDef(opt[In])
      }

      scoptNameAndTermFeatureBagsParams.foreach { optHelper(_) }
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
   * @param paramMap Valid name-and-term generation parameters
   * @return A [[Seq]] of [[String]] representations of the parameters, in a format that can be parsed by Scopt
   */
  def printForCommandLine(paramMap: ParamMap): Seq[String] = {

    NameAndTermFeatureBagsDriver.validateParams(paramMap)

    scoptNameAndTermFeatureBagsParams.flatMap(_.generateCmdLineArgs(paramMap))
  }
}
