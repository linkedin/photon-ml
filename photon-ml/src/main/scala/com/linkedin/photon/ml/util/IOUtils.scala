package com.linkedin.photon.ml.util

import java.io.{BufferedReader, InputStreamReader}

import scala.collection.mutable

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.joda.time.Days
import org.joda.time.format.DateTimeFormat

/**
 * @author xazhang
 */
object IOUtils {

  def getInputPathsWithinDateRange(
      inputDirs: Seq[String],
      startDate: String,
      endDate: String,
      configuration: Configuration,
      errorOnMissing: Boolean): Seq[String] = {

    inputDirs.map(inputDir => getInputPathsWithinDateRange(inputDir, startDate, endDate, configuration, errorOnMissing))
        .reduce(_ ++ _)
  }

  def getInputPathsWithinDateRange(
      inputDirs: String,
      startDate: String,
      endDate: String,
      configuration: Configuration,
      errorOnMissing: Boolean): Seq[String] = {

    val dailyDir = new Path(inputDirs, "daily")
    val from = DateTimeFormat.forPattern("yyyyMMdd").parseLocalDate(startDate)
    val until = DateTimeFormat.forPattern("yyyyMMdd").parseLocalDate(endDate)
    val numberOfDays = Days.daysBetween(from, until).getDays
    val paths = (0 to numberOfDays).map { day =>
      new Path(dailyDir, from.plusDays(day).toString("yyyy/MM/dd"))
    }
    if (errorOnMissing) {
      paths.foreach(path => assert(path.getFileSystem(configuration).exists(path), s"Path $path does not exist!"))
    }
    val existingPaths = paths.filter(path => path.getFileSystem(configuration).exists(path))
    assert(existingPaths.nonEmpty, s"No data folder found between $startDate and $endDate in $dailyDir")
    existingPaths.map(_.toString)
  }

  def readStringsFromHDFS(inputPath: String, configuration: Configuration): mutable.ArrayBuffer[String] = {
    readStringsFromHDFS(new Path(inputPath), configuration)
  }

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

  def writeStringsToHDFS(
      stringMsgs: Iterator[String],
      outputPath: String,
      configuration: Configuration,
      forceOverwrite: Boolean): Unit = {

    writeStringsToHDFS(stringMsgs, new Path(outputPath), configuration, forceOverwrite)
  }

  def writeStringsToHDFS(
      stringMsgs: Iterator[String],
      outputPath: Path,
      configuration: Configuration,
      forceOverwrite: Boolean): Unit = {

    val fs = outputPath.getFileSystem(configuration)
    val stream = fs.create(outputPath, forceOverwrite)
    stringMsgs.foreach(stringMsg => stream.writeBytes(stringMsg + "\n"))
    stream.close()
  }
}
