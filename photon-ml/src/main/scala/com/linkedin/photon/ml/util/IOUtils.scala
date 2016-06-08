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
package com.linkedin.photon.ml.util

import java.io._

import scala.collection.mutable

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.joda.time.Days

/**
 * Some basic IO util functions to be merged with the other util functions
 */
protected[ml] object IOUtils {

  /**
    * Check if the given directory already exists or not
    * @param dir the directory path
    * @param hadoopConf the Hadoop Configuration object
    * @return whether the given directory already exists
    */
  def isDirExisting(dir: String, hadoopConf: Configuration): Boolean = {
    val path = new Path(dir)
    val fs = path.getFileSystem(hadoopConf)
    fs.exists(path)
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
      outputDir: String,
      deleteOutputDirIfExists: Boolean,
      configuration: Configuration): Unit = {

    if (deleteOutputDirIfExists) {
      Utils.deleteHDFSDir(outputDir, configuration)
    } else {
      if (IOUtils.isDirExisting(outputDir, configuration)) {
        throw new IllegalArgumentException(s"Directory $outputDir already exists!")
      }
    }
  }

  /**
    * Returns file paths matching the given date range. This method filters out invalid paths by default, but this
    * behavior can be changed with the "errorOnMissing" parameter.
    *
    * @param inputDirs the base paths for input files
    * @param dateRange date range for finding input files
    * @param configuration Hadoop configuration
    * @param errorOnMissing if true, the method will throw when a date has no corresponding input file
    * @return a sequence of matching file paths
    */
  def getInputPathsWithinDateRange(
      inputDirs: Seq[String],
      dateRange: DateRange,
      configuration: Configuration,
      errorOnMissing: Boolean): Seq[String] = {

    inputDirs.map(inputDir => getInputPathsWithinDateRange(inputDir, dateRange, configuration, errorOnMissing))
        .reduce(_ ++ _)
  }

  /**
   * Returns file paths matching the given date range. This method filters out invalid paths by default, but this
   * behavior can be changed with the "errorOnMissing" parameter.
   *
   * @param inputDirs the base path for input files
   * @param dateRange date range for finding input files
   * @param configuration Hadoop configuration
   * @param errorOnMissing if true, the method will throw when a date has no corresponding input file
   * @return a sequence of matching file paths
   */
  protected def getInputPathsWithinDateRange(
      inputDirs: String,
      dateRange: DateRange,
      configuration: Configuration,
      errorOnMissing: Boolean): Seq[String] = {

    val dailyDir = new Path(inputDirs, "daily")
    val numberOfDays = Days.daysBetween(dateRange.startDate, dateRange.endDate).getDays
    val paths = (0 to numberOfDays).map { day =>
      new Path(dailyDir, dateRange.startDate.plusDays(day).toString("yyyy/MM/dd"))
    }

    if (errorOnMissing) {
      paths.foreach(path => require(path.getFileSystem(configuration).exists(path), s"Path $path does not exist!"))
    }

    val existingPaths = paths.filter(path => path.getFileSystem(configuration).exists(path))
    require(existingPaths.nonEmpty,
      s"No data folder found between ${dateRange.startDate} and ${dateRange.endDate} in $dailyDir")

    existingPaths.map(_.toString)
  }

  /**
    * Read a [[mutable.ArrayBuffer]] of strings from the input path on HDFS
    * @param inputPath the input path
    * @param configuration the Hadoop configuration
    * @return a [[mutable.ArrayBuffer]] of strings read from the input path on HDFS
    */
  def readStringsFromHDFS(inputPath: String, configuration: Configuration): mutable.ArrayBuffer[String] = {
    readStringsFromHDFS(new Path(inputPath), configuration)
  }

  /**
    * Read a [[mutable.ArrayBuffer]] of strings from the input path on HDFS
    * @param inputPath the input path
    * @param configuration the Hadoop configuration
    * @return a [[mutable.ArrayBuffer]] of strings read from the input path on HDFS
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
    * Write an iterator of strings to HDFS
    * @param stringMsgs the strings to be written to HDFS
    * @param outputPath the HDFS path to write the strings
    * @param configuration hadoop configuration
    * @param forceOverwrite whether to force overwrite the output path if already exists
    */
  def writeStringsToHDFS(
      stringMsgs: Iterator[String],
      outputPath: String,
      configuration: Configuration,
      forceOverwrite: Boolean): Unit = {

    writeStringsToHDFS(stringMsgs, new Path(outputPath), configuration, forceOverwrite)
  }

  /**
    * Write an iterator of strings to HDFS
    * @param stringMsgs the strings to be written to HDFS
    * @param outputPath the HDFS path to write the strings
    * @param configuration hadoop configuration
    * @param forceOverwrite whether to force overwrite the output path if already exists
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
        new OutputStreamWriter(stream)
      )
    )
    try {
      stringMsgs.foreach(stringMsg => writer.println(stringMsg))
    } finally {
      writer.close()
    }
  }
}
