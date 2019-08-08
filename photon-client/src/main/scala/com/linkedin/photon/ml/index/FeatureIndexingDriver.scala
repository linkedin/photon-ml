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
package com.linkedin.photon.ml.index

import java.util.{List => JList}

import scala.collection.JavaConverters._

import org.apache.avro.generic.GenericRecord
import org.apache.commons.cli.MissingArgumentException
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkContext}
import org.joda.time.DateTimeZone
import org.slf4j.Logger

import com.linkedin.photon.ml.Types.FeatureShardId
import com.linkedin.photon.ml.data.avro._
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.io.scopt.index.ScoptFeatureIndexingParametersParser
import com.linkedin.photon.ml.util._
import com.linkedin.photon.ml.{Constants, SparkSessionConfiguration}

/**
 * A driver to build feature index maps as an independent Spark job. Recommended when the feature space is large,
 * typically when there are more than 200k unique features.
 */
object FeatureIndexingDriver extends PhotonParams with Logging {

  override val uid = "Feature_Indexing_Driver"
  protected implicit val parent: Identifiable = this

  private val DEFAULT_APPLICATION_NAME = "Feature-Indexing-Job"

  protected[index] var sc: SparkContext = _

  //
  // Parameters
  //

  val inputDataDirectories: Param[Set[Path]] = ParamUtils.createParam(
    "input data directories",
    "Paths to directories containing input data.",
    PhotonParamValidators.nonEmpty[Set, Path])

  val inputDataDateRange: Param[DateRange] = ParamUtils.createParam[DateRange](
    "input data date range",
    "Inclusive date range for input data. If specified, the input directories are expected to be in the daily format " +
      "structure (i.e. trainDir/2017/01/20/[input data files])")

  val inputDataDaysRange: Param[DaysRange] = ParamUtils.createParam[DaysRange](
    "input data days range",
    "Inclusive date range for input data, computed from a range of days prior to today.  If specified, the input " +
      "directories are expected to be in the daily format structure (i.e. trainDir/2017/01/20/[input data files]).")

  val minInputPartitions: Param[Int] = ParamUtils.createParam[Int](
    "minimum input partitions",
    "Minimum number of partitions for the input data.",
    ParamValidators.gt[Int](0.0))

  val rootOutputDirectory: Param[Path] = ParamUtils.createParam[Path](
    "root output directory",
    "Path to base output directory for feature indices.")

  val overrideOutputDirectory: Param[Boolean] = ParamUtils.createParam[Boolean](
    "override output directory",
    "Whether to override the contents of the output directory, if it already exists.")

  val numPartitions: Param[Int] = ParamUtils.createParam[Int](
    "num storage partitions",
    "The number of partitions to break the storage into. This is merely introduced as an optimization at the index " +
      "building stage, so that we don't need to shuffle all features into the same partition. A heuristically good " +
      "number could be 1-10, or even as large as 20; but please avoid setting it into an arbitrarily large number. " +
      "Use just 1 or 2 if you find the job is already fast enough.",
    ParamValidators.gt[Int](0.0))

  val featureShardConfigurations: Param[Map[FeatureShardId, FeatureShardConfiguration]] =
    ParamUtils.createParam[Map[FeatureShardId, FeatureShardConfiguration]](
      "feature shard configurations",
      "A map of feature shard IDs to configurations.",
      PhotonParamValidators.nonEmpty[TraversableOnce, (FeatureShardId, FeatureShardConfiguration)])

  val applicationName: Param[String] = ParamUtils.createParam[String](
    "application name",
    "The name for this Spark application.")

  val timeZone: Param[DateTimeZone] = ParamUtils.createParam[DateTimeZone](
    "time zone",
    "The time zone to use for days ago calculations. See: http://joda-time.sourceforge.net/timezones.html")

  //
  // Initialize object
  //

  setDefaultParams()

  //
  // Params trait extensions
  //

  /**
   * Copy function has no meaning for Driver object. Add extra parameters to params and return.
   *
   * @param extra Additional parameters which should overwrite the values being copied
   * @return This object
   */
  override def copy(extra: ParamMap): Params = {
    extra.toSeq.foreach(set)

    this
  }

  //
  // PhotonParams trait extensions
  //

  /**
   * Set default values for parameters that have them.
   */
  override protected def setDefaultParams(): Unit = {

    setDefault(overrideOutputDirectory, false)
    setDefault(minInputPartitions, 1)
    setDefault(applicationName, DEFAULT_APPLICATION_NAME)
    setDefault(timeZone, Constants.DEFAULT_TIME_ZONE)
  }

  /**
   * Check that all required parameters have been set and validate interactions between parameters.
   *
   * @note In Spark, interactions between parameters are checked by
   *       [[org.apache.spark.ml.PipelineStage.transformSchema()]]. Since we do not use the Spark pipeline API in
   *       Photon-ML, we need to have this function to check the interactions between parameters.
   * @throws MissingArgumentException if a required parameter is missing
   * @throws IllegalArgumentException if a required parameter is missing or a validation check fails
   * @param paramMap The parameters to validate
   */
  override def validateParams(paramMap: ParamMap = extractParamMap): Unit = {

    // Just need to check that these parameters are explicitly set
    paramMap(inputDataDirectories)
    paramMap(rootOutputDirectory)
    paramMap(numPartitions)
    paramMap(featureShardConfigurations)
  }

  //
  // Training driver functions
  //

  /**
   * Run the configured [[FeatureIndexingDriver]].
   */
  def run(): Unit = {

    validateParams()

    // Handle date range input
    val dateRangeOpt = IOUtils.resolveRange(get(inputDataDateRange), get(inputDataDaysRange), getOrDefault(timeZone))
    val inputPaths = dateRangeOpt
      .map { dateRange =>
        IOUtils.getInputPathsWithinDateRange(
          getRequiredParam(inputDataDirectories),
          dateRange,
          sc.hadoopConfiguration,
          errorOnMissing = false)
      }
      .getOrElse(getRequiredParam(inputDataDirectories).toSeq)

    val inputRdd = AvroUtils.readAvroFiles(sc, inputPaths.map(_.toString), getOrDefault(minInputPartitions))

    cleanOutputDir()

    getRequiredParam(featureShardConfigurations).foreach { case (featureShardId, featureShardConfiguration) =>
      val featuresRdd = partitionedUniqueFeatures(
        inputRdd,
        featureShardConfiguration.hasIntercept,
        featureShardConfiguration.featureBags)

      buildIndexMap(featuresRdd, featureShardId)
    }
  }

  /**
   * Ensures that the output path exists.
   */
  private def cleanOutputDir(): Unit = {

    val configuration = sc.hadoopConfiguration
    val outputDir = getRequiredParam(rootOutputDirectory)

    IOUtils.processOutputDir(outputDir, getOrDefault(overrideOutputDirectory), configuration)
    Utils.createHDFSDir(outputDir, configuration)
  }

  /**
   * Given a raw input data [[RDD]], generate the partitioned unique features names grouped by hash code.
   *
   * @param inputRdd An [[RDD]] of raw input [[GenericRecord]]
   * @param addIntercept Whether to add an intercept feature to the list of unique features
   * @param featureSections The set of feature bags to combine into one index
   * @return RDD[(hash key, Iterable[unique feature name])]
   */
  private def partitionedUniqueFeatures(
      inputRdd: RDD[GenericRecord],
      addIntercept: Boolean,
      featureSections: Set[String]): RDD[(Int, Iterable[String])] = {

    val keyedFeaturesRDD = inputRdd
      .flatMap { record: GenericRecord =>
        // Step 1: Extract feature names
        featureSections
          .map { section =>
            Option(record.get(section))
              .getOrElse(throw new IllegalArgumentException(s"Feature section not found: $section"))
              .asInstanceOf[JList[GenericRecord]]
              .asScala
              .toSet
          }
          .reduce(_ ++ _)
          .map { record =>
            Utils.getFeatureKey(record, AvroFieldNames.NAME, AvroFieldNames.TERM, Constants.DELIMITER)
          }
      }.mapPartitions { iter =>
        // Step 2: Map features to (feature hash code, feature name)
        iter.toSet[String].map(f => (f.hashCode, f)).iterator
      }

    val keyedFeaturesUnionedRDD = if (addIntercept) {
      val interceptRDD = sc.parallelize(List[(Int, String)](Constants.INTERCEPT_KEY.hashCode() -> Constants.INTERCEPT_KEY))

      keyedFeaturesRDD.union(interceptRDD)
    } else {
      keyedFeaturesRDD
    }

    // Step 3: Distinct and group by hash code (Note: integer hash code is itself; this trick saves shuffle data size)
    keyedFeaturesUnionedRDD.distinct().groupByKey(new HashPartitioner(getRequiredParam(numPartitions)))
  }

  /**
   * Build and write the index map.
   *
   * @param featuresRdd An RDD of the features
   * @param featureShardId Unique namespace for the index map (default: global)
   */
  private def buildIndexMap(
      featuresRdd: RDD[(Int, Iterable[String])],
      featureShardId: FeatureShardId): Unit = {

    val outputDir = getRequiredParam(rootOutputDirectory).toString
    val projectRdd = featuresRdd.mapPartitionsWithIndex{ case (idx, iter) =>
      var i: Int = 0
      // NOTE PalDB writer within the same JVM might stomp on each other and generate corrupted data, it's safer to
      // lock the write. This will only block writing operations within the same JVM
      PalDBIndexMapBuilder.WRITER_LOCK.synchronized {
        val mapBuilder =
          new PalDBIndexMapBuilder().init(outputDir, idx, featureShardId)

        while (iter.hasNext) {
          val tuple = iter.next()
          val features = tuple._2
          features.foreach { feature =>
            mapBuilder.put(feature, i)
            i += 1
          }
        }

        mapBuilder.close()
      }
      Array[Int](i).toIterator
    }

    // Trigger run
    val num = projectRdd.sum().toInt
    logger.info(s"Total number of features indexed: [$num]")
  }

  /**
   * Entry point to the job.
   *
   * @param args The command line arguments for the job
   */
  def main(args: Array[String]): Unit = {

    // Parse and apply parameters
    val params: ParamMap = ScoptFeatureIndexingParametersParser.parseFromCommandLine(args)
    params.toSeq.foreach(set)

    implicit val log: Logger = logger
    sc = SparkSessionConfiguration.asYarnClient(getOrDefault(applicationName), useKryo = true).sparkContext

    try {

      Timed("Total time in feature indexing driver")(run())

    } catch { case e: Exception =>

      log.error("Failure while running the driver", e)
      throw e

    } finally {

      sc.stop()
    }
  }
}
