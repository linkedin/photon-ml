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
package com.linkedin.photon.ml.avro.data

import java.text.SimpleDateFormat
import java.util.{Calendar, TimeZone}
import scala.collection.{Map, Set}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import scopt.OptionParser

import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.avro.{AvroIOUtils, AvroUtils}
import com.linkedin.photon.ml.util._

/**
 * A class contain [[NameAndTerm]] features sets for each feature section keys
 *
 * @param nameAndTermFeatureSets A [[Map]] of feature section key to [[NameAndTerm]] feature sets
 */
// TODO: Change the scope to [[com.linkedin.photon.ml.avro]] after Avro related classes/functions are decoupled from the
// rest of code
protected[ml] class NameAndTermFeatureSetContainer(nameAndTermFeatureSets: Map[String, Set[NameAndTerm]]) {

  /**
   * Get the map from feature name of type [[NameAndTerm]] to feature index of type [[Int]] based on the specified
   * feature section keys from the input.
   *
   * @param featureSectionKeys The specified feature section keys to generate the name to index map explained above
   * @param isAddingIntercept Whether to add a dummy variable to the generated feature map to learn an intercept term
   * @return The generated map from feature name of type [[NameAndTerm]] to feature index of type [[Int]]
   */
  def getFeatureNameAndTermToIndexMap(
      featureSectionKeys: Set[String],
      isAddingIntercept: Boolean): Map[NameAndTerm, Int] = {

    val featureNameAndTermToIndexMap = nameAndTermFeatureSets.filterKeys(featureSectionKeys.contains).values
        .fold(Set[NameAndTerm]())(_ ++ _).zipWithIndex.toMap
    if (isAddingIntercept) {
      featureNameAndTermToIndexMap +
          (NameAndTerm.INTERCEPT_NAME_AND_TERM -> featureNameAndTermToIndexMap.size)
    } else {
      featureNameAndTermToIndexMap
    }
  }

  /**
   * Write each of the feature map to HDFS.
   *
   * @param nameAndTermFeatureSetContainerOutputDir The HDFS directory to write the feature sets as text files
   * @param sparkContext The Spark context
   */
  def saveAsTextFiles(nameAndTermFeatureSetContainerOutputDir: String, sparkContext: SparkContext): Unit = {
    nameAndTermFeatureSets.foreach { case (featureSectionKey, featureSet) =>
      val featureSetPath = new Path(nameAndTermFeatureSetContainerOutputDir, featureSectionKey)
      NameAndTermFeatureSetContainer.saveNameAndTermSetAsTextFiles(featureSet, sparkContext, featureSetPath)
    }
  }
}

object NameAndTermFeatureSetContainer {

  /**
   * Parse the [[NameAndTermFeatureSetContainer]] from text files on HDFS.
   *
   * @param nameAndTermFeatureSetContainerInputDir The input HDFS directory
   * @param featureSectionKeys The set of feature section keys to look for from the input directory
   * @param configuration The Hadoop configuration
   * @return This [[NameAndTermFeatureSetContainer]] parsed from text files on HDFS
   */
  protected[ml] def readNameAndTermFeatureSetContainerFromTextFiles(
      nameAndTermFeatureSetContainerInputDir: String,
      featureSectionKeys: Set[String],
      configuration: Configuration): NameAndTermFeatureSetContainer = {

    val nameAndTermFeatureSets = featureSectionKeys.map { featureSectionKey =>
      val inputPath = new Path(nameAndTermFeatureSetContainerInputDir, featureSectionKey)
      val nameAndTermFeatureSet = readNameAndTermSetFromTextFiles(inputPath, configuration)
      (featureSectionKey, nameAndTermFeatureSet)
    }.toMap
    new NameAndTermFeatureSetContainer(nameAndTermFeatureSets)
  }

  /**
   * Read a [[Set]] of [[NameAndTerm]] from the text files within the input path.
   *
   * @param inputPath The input path
   * @param configuration the Hadoop configuration
   * @return The [[Set]] of [[NameAndTerm]] read from the text files of the given input path
   */
  private def readNameAndTermSetFromTextFiles(inputPath: Path, configuration: Configuration): Set[NameAndTerm] = {
    IOUtils.readStringsFromHDFS(inputPath, configuration).map { string =>
      string.split("\t") match {
        case Array(name, term) => NameAndTerm(name, term)
        case Array(name) => NameAndTerm(name, "")
        case other => throw new UnsupportedOperationException(
          s"Unexpected entry $string when parsing it to NameAndTerm, " +
              s"after splitting by tab the expected number of tokens is 1 or 2, but found ${other.length}}.")
      }
    }.toSet
  }

  /**
   * Write the [[Set]] of [[NameAndTerm]]s to HDFS as text files.
   *
   * @param nameAndTermSet The map to be written
   * @param sparkContext The Spark context
   * @param outputPath The HDFS path to which write the map
   */
  private def saveNameAndTermSetAsTextFiles(
      nameAndTermSet: Set[NameAndTerm],
      sparkContext: SparkContext,
      outputPath: Path): Unit = {
    val iterator = nameAndTermSet.iterator.map { case NameAndTerm(name, term) => s"$name\t$term" }
    IOUtils.writeStringsToHDFS(iterator, outputPath, sparkContext.hadoopConfiguration, forceOverwrite = false)
  }

  /**
   *
   * @param args
   */
  def main(args: Array[String]): Unit = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Generate-Feature-Name-And-Term-List") {
      head("Generate the nameAndTerm to index feature map.")
      opt[String]("data-input-directory")
          .required()
          .text("input directories of data to be processed in response prediction AVRO format. " +
          "Multiple input directories are separated by commas.")
          .action((x, c) => c.copy(inputDirs = x.split(",")))
      opt[String]("date-range")
          .text(s"date range for the input data represented in the form start.date-end.date, e.g. 20150501-20150631, " +
          s"default: ${defaultParams.dateRangeOpt}")
          .action((x, c) => c.copy(dateRangeOpt = Some(x)))
      opt[String]("date-range-days-ago")
          .text(s"date range for the input data represented in the form start.daysAgo-end.daysAgo, e.g. 90-1, " +
          s"default: ${defaultParams.dateRangeDaysAgoOpt}")
          .action((x, c) => c.copy(dateRangeDaysAgoOpt = Some(x)))
      opt[Int]("num-days-data-for-feature-generation")
          .text(s"Number of days of data used for feature generation. Currently this parameter is only used in " +
          s"the weekly/monthly feature generation pipeline. If date-range is specified, the input of this option " +
          s"will be ignored. Default: ${defaultParams.numDaysDataForFeatureGeneration}.")
          .action((x, c) => c.copy(numDaysDataForFeatureGeneration = x))
      opt[String]("feature-name-and-term-set-output-dir").required()
          .text(s"output path for the generated feature NameAndTerm set")
          .action((x, c) => c.copy(featureNameAndTermSetOutputPath = x))
      opt[String]("feature-section-keys")
          .text(s"Comma separated ist of feature section keys used to generate the feature NameAndTerm set")
          .action((x, c) => c.copy(featureSectionKeys = x.split(",").toSet))
      opt[Boolean]("delete-output-dir-if-exists")
          .text(s"Whether to delete the output directory if exists. Default: ${defaultParams.deleteOutputDirIfExists}")
          .action((x, c) => c.copy(deleteOutputDirIfExists = x))
      opt[String]("application-name")
          .text(s"Name of this Spark application, ${defaultParams.applicationName}")
          .action((x, c) => c.copy(applicationName = x))
      help("help").text("prints usage text")
    }
    val params = parser.parse(args, Params()) match {
      case Some(parsedParams) => parsedParams
      case None => throw new IllegalArgumentException(s"Parsing the command line arguments failed " +
          s"(${args.mkString(", ")}),\n ${parser.usage}")
    }
    import params._

    println(params + "\n")
    val sparkContext = SparkContextConfiguration.asYarnClient(applicationName, useKryo = true)
    val configuration = sparkContext.hadoopConfiguration
    // Process the output directory upfront and potentially fail the job early
    IOUtils.processOutputDir(featureNameAndTermSetOutputPath, deleteOutputDirIfExists, configuration)

    println(s"Application applicationName: $applicationName")

    // If date-range is specified, this parameter will be ignored.
    val adjustedDateRangeOpt = dateRangeOpt match {
      case Some(dateRange) => Some(dateRange)
      case None =>
        if (numDaysDataForFeatureGeneration < Int.MaxValue) {
          val dailyPlainFormat = new SimpleDateFormat("yyyyMMdd")
          dailyPlainFormat.setTimeZone(TimeZone.getTimeZone("America/Los_Angeles"))
          val calendar = Calendar.getInstance()
          // The latest training data is yesterday's data
          calendar.add(Calendar.DATE, -1)
          val yesterdayDate = calendar.getTime
          // Backtracking to get the starting date of the training data
          calendar.add(Calendar.DATE, -(1 + numDaysDataForFeatureGeneration))
          Some(s"${dailyPlainFormat.format(calendar.getTime)}-${dailyPlainFormat.format(yesterdayDate)}")
        } else {
          None
        }
    }

    val inputRecordsPath = (adjustedDateRangeOpt, dateRangeDaysAgoOpt) match {
      // Specified as date range
      case (Some(dateRange), None) =>
        val range = DateRange.fromDates(dateRange)
        IOUtils.getInputPathsWithinDateRange(inputDirs, range, sparkContext.hadoopConfiguration,
          errorOnMissing = false)

      // Specified as a range of start days ago - end days ago
      case (None, Some(dateRangeDaysAgo)) =>
        val range = DateRange.fromDaysAgo(dateRangeDaysAgo)
        IOUtils.getInputPathsWithinDateRange(inputDirs, range, sparkContext.hadoopConfiguration,
          errorOnMissing = false)

      // Both types specified: illegal
      case (Some(_), Some(_)) =>
        throw new IllegalArgumentException(
          "Both dateRangeOpt and dateRangeDaysAgoOpt given. You must specify date ranges using only one " +
          "format.")

      case (None, None) => inputDirs.toSeq
    }
    println(s"inputRecordsPath:\n${inputRecordsPath.mkString("\n")}")

    val numExecutors = sparkContext.getExecutorStorageStatus.length
    val minPartitions =
      if (sparkContext.getConf.contains("spark.default.parallelism")) {
        sparkContext.defaultParallelism
      } else {
        numExecutors * 5
      }
    val records = AvroIOUtils.readAvroFiles(sparkContext, inputRecordsPath, minPartitions)
    // numExecutors * 5 is too much for distinct operation when the data are huge. Use numExecutors instead.
    val nameAndTermFeatureSetContainer =
      AvroUtils.readNameAndTermFeatureSetContainerFromGenericRecords(records, featureSectionKeys, numExecutors)
    nameAndTermFeatureSetContainer.saveAsTextFiles(featureNameAndTermSetOutputPath, sparkContext)

    sparkContext.stop()
  }

  private case class Params(
      inputDirs: Array[String] = Array(),
      dateRangeOpt: Option[String] = None,
      dateRangeDaysAgoOpt: Option[String] = None,
      numDaysDataForFeatureGeneration: Int = Int.MaxValue,
      featureNameAndTermSetOutputPath: String = "",
      featureSectionKeys: Set[String] = Set(),
      deleteOutputDirIfExists: Boolean = false,
      applicationName: String = "Generate-name-and-term-feature-set") {

    /**
     *
     * @return
     */
    override def toString: String = {
      s"Input parameters:\n" +
          s"inputDirs: ${inputDirs.mkString(", ")}\n" +
          s"dateRangeOpt: $dateRangeOpt\n" +
          s"dateRangeDaysAgoOpt: $dateRangeDaysAgoOpt\n" +
          s"numDaysDataForFeatureGeneration: $numDaysDataForFeatureGeneration\n" +
          s"featureNameAndTermSetOutputPath:\n$featureNameAndTermSetOutputPath\n" +
          s"featureSectionKeys: ${featureSectionKeys.mkString(", ")}\n" +
          s"deleteOutputDirIfExists: $deleteOutputDirIfExists\n" +
          s"applicationName: $applicationName"
    }
  }
}
