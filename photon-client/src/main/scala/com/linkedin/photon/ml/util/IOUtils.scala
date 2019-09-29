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
package com.linkedin.photon.ml.util

import java.io._

import scala.collection.mutable
import scala.util.Try

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileContext, Options, Path}
import org.apache.spark.SparkContext
import org.joda.time.{DateTimeZone, Days}

import com.linkedin.photon.ml.Constants
import com.linkedin.photon.ml.estimators.GameEstimator
import com.linkedin.photon.ml.index.IndexMapLoader
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Some basic IO util functions to be merged with the other util functions.
 */
object IOUtils {

  /**
   * Resolve between multiple date range specification options.
   *
   * @param dateRangeOpt Optional date range specified using a [[DateRange]]
   * @param daysRangeOpt Optional date range specified using a [[DaysRange]]
   * @param timeZone The local timezone to use for date math
   * @return A single [[DateRange]] to use, if either of the date range options are specified
   * @throws IllegalArgumentException If both date ranges are specified
   */
  def resolveRange(
      dateRangeOpt: Option[DateRange],
      daysRangeOpt: Option[DaysRange],
      timeZone: DateTimeZone = Constants.DEFAULT_TIME_ZONE): Option[DateRange] =

    (dateRangeOpt, daysRangeOpt) match {

      // Specified as date range
      case (Some(dateRange), None) => Some(dateRange)

      // Specified as a range of start days ago - end days ago
      case (None, Some(daysRange)) => Some(daysRange.toDateRange(timeZone))

      // Both types specified: illegal
      case (Some(_), Some(_)) =>
        throw new IllegalArgumentException(
          "Both date range and days ago given. You must specify date ranges using only one format.")

      // No range specified, just use the train dir
      case (None, None) => None
    }

  /**
   * Check if the given directory already exists or not.
   *
   * @param dir The directory path
   * @param hadoopConf The Hadoop Configuration object
   * @return Whether the given directory already exists
   */
  def isDirExisting(dir: Path, hadoopConf: Configuration): Boolean = {
    val fs = dir.getFileSystem(hadoopConf)
    fs.exists(dir)
  }

  /**
   * Process the output directory. If deleteOutputDirIfExists is true, then the output directory will be deleted.
   * Otherwise, an [[IllegalArgumentException]] will be thrown if the output directory already exists.
   *
   * @param outputDir The specified output directory
   * @param deleteOutputDirIfExists Whether the output directory should be deleted if exists
   * @param configuration The Hadoop Configuration object
   */
  protected[ml] def processOutputDir(
      outputDir: Path,
      deleteOutputDirIfExists: Boolean,
      configuration: Configuration): Unit = {

    if (deleteOutputDirIfExists) {
      Utils.deleteHDFSDir(outputDir, configuration)
    } else {
      if (isDirExisting(outputDir, configuration)) {
        throw new IllegalArgumentException(s"Directory $outputDir already exists")
      }
    }
  }

  /**
   * Returns file paths matching the given date range. This method filters out invalid paths by default, but this
   * behavior can be changed with the "errorOnMissing" parameter.
   *
   * @param inputDirs The base paths for input files
   * @param dateRange Date range for finding input files
   * @param configuration Hadoop configuration
   * @param errorOnMissing If true, the method will throw when a date has no corresponding input file
   * @return A sequence of matching file paths
   */
  def getInputPathsWithinDateRange(
      inputDirs: Set[Path],
      dateRange: DateRange,
      configuration: Configuration,
      errorOnMissing: Boolean): Seq[Path] =
    inputDirs
      .map(inputDir => getInputPathsWithinDateRange(inputDir, dateRange, configuration, errorOnMissing))
      .reduce(_ ++ _)

  /**
   * Returns file paths matching the given date range. This method filters out invalid paths by default, but this
   * behavior can be changed with the "errorOnMissing" parameter.
   *
   * @param baseDir The base path for input files
   * @param dateRange Date range for finding input files
   * @param configuration Hadoop configuration
   * @param errorOnMissing If true, the method will throw when a date has no corresponding input file
   * @return A sequence of matching file paths
   */
  protected def getInputPathsWithinDateRange(
      baseDir: Path,
      dateRange: DateRange,
      configuration: Configuration,
      errorOnMissing: Boolean): Seq[Path] = {

    val numberOfDays = Days.daysBetween(dateRange.startDate, dateRange.endDate).getDays
    val paths = (0 to numberOfDays).map { day =>
      new Path(baseDir, dateRange.startDate.plusDays(day).toString("yyyy/MM/dd"))
    }

    if (errorOnMissing) {
      paths.foreach(path => require(path.getFileSystem(configuration).exists(path), s"Path $path does not exist"))
    }

    val existingPaths = paths.filter(path => path.getFileSystem(configuration).exists(path))

    require(
      existingPaths.nonEmpty,
      s"No data folder found between ${dateRange.startDate} and ${dateRange.endDate} in $baseDir")

    existingPaths
  }

  /**
   * Read a [[mutable.ArrayBuffer]] of strings from the input path on HDFS
   *
   * @param inputPath The input path
   * @param configuration The Hadoop configuration
   * @return A [[mutable.ArrayBuffer]] of strings read from the input path on HDFS
   */
  def readStringsFromHDFS(inputPath: String, configuration: Configuration): mutable.ArrayBuffer[String] = {
    readStringsFromHDFS(new Path(inputPath), configuration)
  }

  /**
   * Read a [[mutable.ArrayBuffer]] of strings from the input path on HDFS
   *
   * @param inputPath The input path
   * @param configuration The Hadoop configuration
   * @return A [[mutable.ArrayBuffer]] of strings read from the input path on HDFS
   */
  def readStringsFromHDFS(inputPath: Path, configuration: Configuration): mutable.ArrayBuffer[String] = {
    val fs = inputPath.getFileSystem(configuration)
    val bufferedReader = new BufferedReader(new InputStreamReader(fs.open(inputPath)))
    val arrayBuffer = new mutable.ArrayBuffer[String]
    var line = bufferedReader.readLine()
    while (line != null) {
      arrayBuffer += line
      line = bufferedReader.readLine()
    }
    bufferedReader.close()
    arrayBuffer
  }

  /**
   * Write an [[GameEstimator.GameOptimizationConfiguration]] to HDFS.
   *
   * @param optimizationConfig The GAME model optimization configuration to write to HDFS
   * @param outputPath The output HDFS directory to which to write
   * @param configuration The HDFS configuration
   * @param forceOverwrite Whether to force overwrite the output path if already exists
   */
  def writeOptimizationConfigToHDFS(
      optimizationConfig: GameEstimator.GameOptimizationConfiguration,
      outputPath: Path,
      configuration: Configuration,
      forceOverwrite: Boolean): Unit =
    writeStringsToHDFS(
      Iterator(optimizationConfigToString(optimizationConfig)),
      outputPath,
      configuration,
      forceOverwrite)

  /**
   * Write an iterator of strings to HDFS
   *
   * @param stringMsgs The strings to be written to HDFS
   * @param outputPath The HDFS path to write the strings
   * @param configuration Hadoop configuration
   * @param forceOverwrite Whether to force overwrite the output path if already exists
   */
  def writeStringsToHDFS(
      stringMsgs: Iterator[String],
      outputPath: Path,
      configuration: Configuration,
      forceOverwrite: Boolean): Unit = {

    val fs = outputPath.getFileSystem(configuration)
    val stream = fs.create(outputPath, forceOverwrite)
    val writer = new PrintWriter(
      new BufferedWriter(
        new OutputStreamWriter(stream, "UTF-8")
      )
    )
    try {
      stringMsgs.foreach(stringMsg => writer.println(stringMsg))
    } finally {
      writer.close()
    }
  }

  /**
   * Write a map of learned [[GeneralizedLinearModel]] to text files.
   *
   * @param sc The Spark context
   * @param models The map of (Model Id -> [[GeneralizedLinearModel]])
   * @param modelDir The directory for the output text files
   */
  def writeModelsInText(
      sc: SparkContext,
      models: Iterable[(Double, GeneralizedLinearModel)],
      modelDir: String,
      indexMapLoader: IndexMapLoader): Unit = {

    models.foreach{ case (lambda, m) => println(lambda + ": " + m.toString)}

    sc.parallelize(models.toSeq, models.size)
      .mapPartitions({ iter =>
        val indexMap = indexMapLoader.indexMapForRDD()
        val modelStrs = new mutable.ArrayBuffer[String]()

        while (iter.hasNext) {
          val t = iter.next()
          val regWeight = t._1
          val model = t._2
          val builder = new mutable.ArrayBuffer[String]()

          model.coefficients
            .means
            .toArray
            .zipWithIndex
            .sortWith((p1, p2) => p1._1 > p2._1)
            .foreach { case (value, index) =>
              val nameAndTerm = indexMap.getFeatureName(index)
              nameAndTerm.foreach { s =>
                val tokens = s.split(Constants.DELIMITER)
                if (tokens.length == 1) {
                  builder += s"${tokens(0)}\t${""}\t$value\t$regWeight"
                } else if (tokens.length == 2) {
                  builder += s"${tokens(0)}\t${tokens(1)}\t$value\t$regWeight"
                } else {
                  throw new IOException(s"unknown name and terms: $s")
                }
              }
            }

          val s = builder.mkString("\n")
          modelStrs += s
        }

        modelStrs.iterator
      })
      .saveAsTextFile(modelDir)
  }

  /**
   * Write to a stream while handling exceptions, and closing the stream correctly whether writing to it
   * succeeded or not.
   *
   * @note remember that a Try instance can be understood as a collection, that can have zero
   * or one element. This code uses a "monadic flow" started by the Try. Try can be a Success or a Failure.
   * Success.map(lambda) applies lambda to the value wrapped in the Success instance, and returns the result,
   * which can itself be either Success or Failure, wrapping an instance of the type returned by the lambda.
   * Failure.map(lambda) ignores lambda, and returns itself, but changing the contained type to the type
   * returned by the lambda (see scala.util.Try). Failure thus contains an exception, if one is thrown.
   *
   * @param outputStreamGenerator A lambda that generates an output stream
   * @param op A lambda that writes to the stream
   * @return Success or Failure. In case of Failure, the Failure contains the exception triggered
   */
  def toStream(outputStreamGenerator: => OutputStream)(op: PrintWriter => Unit): Try[Unit] = {

    val os = Try(outputStreamGenerator)
    val writer = os.map(stream => new PrintWriter(stream))

    val write = writer.map(op(_))
    val flush = writer.map(_.flush)
    val close = os.map(_.close)

    write.flatMap(_ => flush).flatMap(_ => close)
  }

  /**
   * Backup and update a file on HDFS.
   *
   * A temporary file is written to, using the writeOp lambda. Then the old file is atomically backed up
   * to a file with the same name and suffix ".prev". Finally, the newly written file is atomically
   * renamed. If any operation in the process fails, the remaining operations are not executed, and an
   * exception is propagated instead.
   *
   * @param sc The Spark context
   * @param fileName The name of the file to backup and update
   * @param writeOp A lambda that writes to the file
   * @return Success or Failure. In case of Failure, the Failure contains the exceptions triggered
   */
  def toHDFSFile(sc: SparkContext, fileName: String)(writeOp: PrintWriter => Unit): Try[Unit] = {

    val cf = sc.hadoopConfiguration
    val (fs, fc) = (org.apache.hadoop.fs.FileSystem.get(cf), FileContext.getFileContext(cf))
    val (file, tmpFile, bkpFile) = (new Path(fileName), new Path(fileName + "-tmp"), new Path(fileName + ".prev"))

    toStream(fs.create(tmpFile))(writeOp)
      .map(_ => if (fs.exists(file)) fc.rename(file, bkpFile, Options.Rename.OVERWRITE))
      .map(_ => fc.rename(tmpFile, file, Options.Rename.OVERWRITE))
  }

  /**
   * Summarize a [[GameEstimator.GameOptimizationConfiguration]] into a human-readable [[String]].
   *
   * @param config A [[GameEstimator.GameOptimizationConfiguration]]
   * @return The summarized config in human-readable text
   */
  def optimizationConfigToString(config: GameEstimator.GameOptimizationConfiguration): String = {

    val builder = new StringBuilder

    config
      .toSeq
      .sortBy { case (coordinateId, coordinateConfig) =>
        val priority = coordinateConfig match {
          case _: FixedEffectOptimizationConfiguration => 1
          case _: RandomEffectOptimizationConfiguration => 2
          case _ =>
            throw new IllegalArgumentException(
              s"Unknown optimization configuration for coordinate $coordinateId with type ${coordinateConfig.getClass}")
        }

        (priority, coordinateId)
      }
      .foreach { case (coordinateId, coordinateConfig) =>
        builder.append(s"$coordinateId:\n$coordinateConfig\n")
      }

    builder.mkString
  }
}
