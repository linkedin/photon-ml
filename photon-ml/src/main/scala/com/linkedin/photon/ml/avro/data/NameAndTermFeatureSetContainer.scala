package com.linkedin.photon.ml.avro.data

import java.text.SimpleDateFormat
import java.util.{Calendar, List => JList, TimeZone}

import scala.collection.{Map, Set, mutable}

import org.apache.avro.generic.GenericRecord
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scopt.OptionParser

import com.linkedin.photon.ml.avro.AvroUtils
import com.linkedin.photon.ml.util._


/**
 * A class contain [[NameAndTerm]] features sets for each feature section keys
 * @param nameAndTermFeatureSets A [[Map]] of feature section key to [[NameAndTerm]] feature sets
 * @author xazhang
 */
protected[photon] class NameAndTermFeatureSetContainer(nameAndTermFeatureSets: Map[String, Set[NameAndTerm]]) {

  def getFeatureNameAndTermToIndexMap(featureSectionKeys: Set[String], isAddingIntercept: Boolean)
  : Map[NameAndTerm, Int] = {

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
   * Write each of the feature map to HDFS
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

protected[photon] object NameAndTermFeatureSetContainer {

  /**
   * Generate the [[NameAndTermFeatureSetContainer]] from a [[RDD]] of [[GenericRecord]]s.
   * @param genericRecords The input [[RDD]] of [[GenericRecord]]s.
   * @param featureSectionKeys The set of feature section keys of interest in the input generic records
   * @return The generated [[NameAndTermFeatureSetContainer]]
   */
  def generateFromGenericRecords(
      genericRecords: RDD[GenericRecord],
      featureSectionKeys: Set[String]): NameAndTermFeatureSetContainer = {

    val nameAndTermFeatureSets = featureSectionKeys.map { featureSectionKey =>
      (featureSectionKey, parseNameAndTermSetFromGenericRecords(genericRecords, featureSectionKey))
    }.toMap

    new NameAndTermFeatureSetContainer(nameAndTermFeatureSets)
  }

  private def parseNameAndTermSetFromGenericRecords(genericRecords: RDD[GenericRecord],
      featureSectionKey: String): Set[NameAndTerm] = {

    genericRecords.flatMap(_.get(featureSectionKey) match { case recordList: JList[_] =>
      val nnz = recordList.size
      val featureNameAndTermBuf = new mutable.ArrayBuffer[NameAndTerm](nnz)
      val iterator = recordList.iterator
      while (iterator.hasNext) {
        iterator.next match {
          case record: GenericRecord =>
            val featureNameAndTerm = AvroUtils.getNameAndTermFromAvroRecord(record)
            featureNameAndTermBuf += featureNameAndTerm
          case any => throw new IllegalArgumentException(s"$any in features list is not a record")
        }
      }
      featureNameAndTermBuf
    case _ => throw new IllegalArgumentException(s"$featureSectionKey is not a list (and might be null)")
    }).distinct().collect().toSet
  }

  /**
   * Parse the [[NameAndTermFeatureSetContainer]] from text files on HDFS
   * @param nameAndTermFeatureSetContainerInputDir The input HDFS directory
   * @param featureSectionKeys The set of feature section keys to look for from the input directory
   * @param configuration The Hadoop configuration
   * @return This [[NameAndTermFeatureSetContainer]] parsed from text files on HDFS
   */
  def loadFromTextFiles(
      nameAndTermFeatureSetContainerInputDir: String,
      featureSectionKeys: Set[String],
      configuration: Configuration): NameAndTermFeatureSetContainer = {

    val nameAndTermFeatureSets = featureSectionKeys.map { featureSectionKey =>
      val inputPath = new Path(nameAndTermFeatureSetContainerInputDir, featureSectionKey)
      val nameAndTermFeatureSet = loadNameAndTermSetFromHDFSPath(inputPath, configuration)
      (featureSectionKey, nameAndTermFeatureSet)
    }.toMap
    new NameAndTermFeatureSetContainer(nameAndTermFeatureSets)
  }

  private def loadNameAndTermSetFromHDFSPath(
      inputPath: Path,
      configuration: Configuration): Set[NameAndTerm] = {
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
    val sparkContext = SparkContextConfiguration.asYarnClient(applicationName)

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

    val inputRecordsPath = adjustedDateRangeOpt match {
      case Some(dateRange) =>
        val Array(startDate, endDate) = dateRange.split("-")
        IOUtils.getInputPathsWithinDateRange(inputDirs, startDate, endDate, sparkContext.hadoopConfiguration,
          errorOnMissing = false)
      case None => inputDirs.toSeq
    }
    println(s"inputRecordsPath:\n${inputRecordsPath.mkString("\n")}")
    val numExecutors = sparkContext.getExecutorStorageStatus.length
    val minPartitions =
      if (sparkContext.getConf.contains("spark.default.parallelism")) {
        sparkContext.defaultParallelism
      } else {
        numExecutors * 5
      }
    val records = AvroUtils.readAvroFiles(sparkContext, inputRecordsPath, minPartitions)
    val nameAndTermFeatureSetContainer = generateFromGenericRecords(records, featureSectionKeys)
    Utils.deleteHDFSDir(featureNameAndTermSetOutputPath, sparkContext.hadoopConfiguration)
    nameAndTermFeatureSetContainer.saveAsTextFiles(featureNameAndTermSetOutputPath, sparkContext)

    sparkContext.stop()
  }

  private case class Params(
      inputDirs: Array[String] = Array(),
      dateRangeOpt: Option[String] = None,
      numDaysDataForFeatureGeneration: Int = Int.MaxValue,
      featureNameAndTermSetOutputPath: String = "",
      featureSectionKeys: Set[String] = Set(),
      applicationName: String = "Generate-name-and-term-feature-set") {

    override def toString = {
      s"Input parameters:\n" +
          s"inputDirs: ${inputDirs.mkString(", ")}\n" +
          s"dateRangeOpt: $dateRangeOpt\n" +
          s"numDaysDataForFeatureGeneration: $numDaysDataForFeatureGeneration\n" +
          s"featureNameAndTermSetOutputPath:\n$featureNameAndTermSetOutputPath\n" +
          s"featureSectionKeys: ${featureSectionKeys.mkString(", ")}\n" +
          s"applicationName: $applicationName"
    }
  }

}
