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
package com.linkedin.photon.ml.cli.game

import org.apache.commons.cli.MissingArgumentException
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.storage.StorageLevel
import org.joda.time.DateTimeZone
import org.slf4j.Logger

import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.Types.FeatureShardId
import com.linkedin.photon.ml.data.InputColumnsNames
import com.linkedin.photon.ml.data.avro.NameAndTermFeatureMapUtils
import com.linkedin.photon.ml.evaluation.EvaluatorType
import com.linkedin.photon.ml.index.{DefaultIndexMapLoader, IndexMapLoader, PalDBIndexMapLoader}
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.util.Implicits._
import com.linkedin.photon.ml.util._

/**
 * Contains common parameters and functions for GAME training/scoring drivers.
 */
trait GameDriver extends PhotonParams {

  //
  // Members
  //

  val LOGS_FILE_NAME = "logs.txt"

  override val uid = "GAME_Driver"

  protected def sc: SparkContext

  protected def logger: Logger

  protected implicit val parent: Identifiable = this

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

  val offHeapIndexMapDirectory: Param[Path] = ParamUtils.createParam[Path](
    "off-heap index map directory",
    "Path to the directory containing the off-heap feature index map.")

  val offHeapIndexMapPartitions: Param[Int] = ParamUtils.createParam(
    "off-heap index map partitions",
    "Number of partitions for the off-heap feature index map. This number must match the number of partitions that " +
      "the index map was constructed with.",
    ParamValidators.gt[Int](0.0))

  val inputColumnNames: Param[InputColumnsNames] = ParamUtils.createParam[InputColumnsNames](
    "input column names",
    "A map of custom column names which replace the default column names of expected fields in the Avro input.")

  val modelInputDirectory: Param[Path] = ParamUtils.createParam(
    "model input directory",
    "Path to directory containing GAME model.")

  val evaluators: Param[Seq[EvaluatorType]] = ParamUtils.createParam(
    "validation evaluators",
    "A list of evaluators used to validate computed scores.",
    PhotonParamValidators.nonEmpty[Seq, EvaluatorType])

  val rootOutputDirectory: Param[Path] = ParamUtils.createParam[Path](
    "root output directory",
    "Path to base output directory for logs, scores, trained models, etc.")

  val overrideOutputDirectory: Param[Boolean] = ParamUtils.createParam[Boolean](
    "override output directory",
    "Whether to override the contents of the output directory, if it already exists.")

  val outputFilesLimit: Param[Int] = ParamUtils.createParam[Int](
    "output files limit",
    "The maximum number of output files to write. Tuning parameter to prevent hitting HDFS file limit quota.",
    ParamValidators.gt[Int](0.0))

  val featureBagsDirectory: Param[Path] = ParamUtils.createParam[Path](
    "feature bags directory",
    "Path to the directory containing whitelists of features to use from each feature bag.")

  val featureShardConfigurations: Param[Map[FeatureShardId, FeatureShardConfiguration]] =
    ParamUtils.createParam[Map[FeatureShardId, FeatureShardConfiguration]](
      "feature shard configurations",
      "A map of feature shard IDs to configurations.",
      PhotonParamValidators.nonEmpty[TraversableOnce, (FeatureShardId, FeatureShardConfiguration)])

  val dataValidation: Param[DataValidationType] = ParamUtils.createParam[DataValidationType](
    "data validation",
    "Type of data validation to perform on input data.")

  val logLevel: Param[Int] = ParamUtils.createParam[Int](
    "logging level",
    "The logging level for the output Photon-ML logs.",
    { level: Int => PhotonLogger.logLevelNames.values.toSet.contains(level) })

  val applicationName: Param[String] = ParamUtils.createParam[String](
    "application name",
    "The name for this Spark application.")

  val timeZone: Param[DateTimeZone] = ParamUtils.createParam[DateTimeZone](
    "time zone",
    "The time zone to use for days ago calculations. See: http://joda-time.sourceforge.net/timezones.html")

  //
  // PhotonParams trait extensions
  //

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
    paramMap(featureShardConfigurations)

    (paramMap.get(offHeapIndexMapDirectory), paramMap.get(offHeapIndexMapPartitions)) match {
      case (Some(_), None) =>
        throw new IllegalArgumentException(
          "Off-heap index map directory provided without off-heap index map partitions.")

      case (None, Some(_)) =>
        throw new IllegalArgumentException(
          "Off-heap index map partitions provided without off-heap index map directory.")

      case _ =>
    }

    (paramMap.get(offHeapIndexMapDirectory), paramMap.get(featureBagsDirectory)) match {
      case (Some(_), Some(_)) =>
        throw new IllegalArgumentException(
          "Ambiguous index map input: both off-heap index map directory and feature bags directory provided")

      case _ =>
    }
  }

  //
  // Common driver functions
  //

  /**
   * Builds feature key to index map loaders according to configuration.
   * Also, sets intercept terms ON by default.
   *
   * @return A map of shard id to feature map loader
   * @deprecated This function will be removed in the next major version.
   */
  protected[game] def prepareFeatureMapsDefault(): Map[FeatureShardId, IndexMapLoader] = {

    val shardConfigs = getRequiredParam(featureShardConfigurations)
    val allFeatureBagKeys = shardConfigs.values.map(_.featureBags).reduce(_ ++ _)
    val nameAndTermFeatureMap = NameAndTermFeatureMapUtils.readNameAndTermFeatureMapFromTextFiles(
      getRequiredParam(featureBagsDirectory),
      allFeatureBagKeys,
      sc)
    // persist [[NameAndTerm]] feature [[RDD]]'s as they may be used multiple times below
    nameAndTermFeatureMap.foreach { case (_, v) => v.persist(StorageLevel.MEMORY_ONLY_SER) }
    val featureShardIdToFeatureMapLoader = shardConfigs.map { case (shardId, featureShardConfig) =>
      val featureMap = NameAndTermFeatureMapUtils
        .getFeatureNameAndTermToIndexMap(
          nameAndTermFeatureMap, featureShardConfig.featureBags, featureShardConfig.hasIntercept, sc)
        .map { case (k, v) => Utils.getFeatureKey(k.name, k.term) -> v }
      val indexMapLoader = new DefaultIndexMapLoader(sc, featureMap)

      (shardId, indexMapLoader)
    }
    nameAndTermFeatureMap.foreach { case (_, v) => v.unpersist() }

    featureShardIdToFeatureMapLoader.tap { case (shardId, featureMapLoader) =>
      logger.debug(s"Feature shard ID: $shardId, number of features: ${featureMapLoader.indexMapForDriver().size}")
    }
  }

  /**
   * Builds PalDB off-heap feature name-and-term to index map loaders according to configuration.
   *
   * @return A map of shard id to feature map
   */
  protected[game] def prepareFeatureMapsPalDB(): Map[FeatureShardId, IndexMapLoader] =
    getRequiredParam(featureShardConfigurations).map { case (shardId, _) =>
      (shardId, PalDBIndexMapLoader(sc, get(offHeapIndexMapDirectory).get, get(offHeapIndexMapPartitions).get, shardId))
    }

  /**
   * Builds feature name-and-term to index maps according to configuration.
   *
   * @return A map of shard id to feature map
   */
  protected[game] def prepareFeatureMaps(): Option[Map[FeatureShardId, IndexMapLoader]] =
    (get(offHeapIndexMapDirectory), get(featureBagsDirectory)) match {
      // If an off-heap map path is specified, use the PalDB loader
      case (Some(_), None) => Some(prepareFeatureMapsPalDB())
      // Otherwise, fall back to the default loader
      case (None, Some(_)) => Some(prepareFeatureMapsDefault())

      case _ => None
    }

  /**
   * Resolves paths for specified date range to physical file paths.
   *
   * @param baseDirs The base dirs to which date-specific relative paths will be appended
   * @param dateRangeOpt Optional date range
   * @return All resolved paths
   */
  protected[game] def pathsForDateRange(baseDirs: Set[Path], dateRangeOpt: Option[DateRange]): Seq[Path] =
    dateRangeOpt match {
      // Specified as date range
      case Some(dateRange) =>
        IOUtils.getInputPathsWithinDateRange(baseDirs, dateRange, sc.hadoopConfiguration, errorOnMissing = false)

      // No range specified, just use the train dir
      case None => baseDirs.toSeq
    }
}
