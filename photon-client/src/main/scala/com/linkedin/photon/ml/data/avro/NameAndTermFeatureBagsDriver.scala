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
package com.linkedin.photon.ml.data.avro

import org.apache.commons.cli.MissingArgumentException
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.Identifiable
import org.joda.time.DateTimeZone

import com.linkedin.photon.ml.{Constants, SparkSessionConfiguration}
import com.linkedin.photon.ml.io.scopt.avro.ScoptNameAndTermFeatureBagsParametersParser
import com.linkedin.photon.ml.util._

/**
 * A driver to build one or more feature bags from an input data set. These feature bags can be modified to act as
 * feature whitelists for GAME input data.
 */
object NameAndTermFeatureBagsDriver extends PhotonParams with Logging {

  override val uid = "Name_And_Term_Feature_Bags_Driver"
  protected implicit val parent: Identifiable = this

  private val DEFAULT_APPLICATION_NAME = "Name-And-Term-Feature-Bags-Job"
  private val PARALLELISM_MODIFIER = 5

  protected[avro] var sc: SparkContext = _

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

  val rootOutputDirectory: Param[Path] = ParamUtils.createParam[Path](
    "root output directory",
    "Path to base output directory for feature indices.")

  val overrideOutputDirectory: Param[Boolean] = ParamUtils.createParam[Boolean](
    "override output directory",
    "Whether to override the contents of the output directory, if it already exists.")

  val featureBagsKeys: Param[Set[String]] = ParamUtils.createParam[Set[String]](
    "feature bags keys",
    "List of feature bags keys used to generate the feature NameAndTerm set.",
    PhotonParamValidators.nonEmpty[Set, String])

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
    paramMap(featureBagsKeys)
  }

  //
  // Training driver functions
  //

  /**
   * Run the configured [[NameAndTermFeatureBagsDriver]].
   */
  def run(): Unit = {

    validateParams()
    cleanOutputDir()

    // Handle date range input
    val dateRangeOpt = IOUtils.resolveRange(get(inputDataDateRange), get(inputDataDaysRange), getOrDefault(timeZone))
    val inputPathsWithRanges = dateRangeOpt
      .map { dateRange =>
        IOUtils.getInputPathsWithinDateRange(
          getRequiredParam(inputDataDirectories),
          dateRange,
          sc.hadoopConfiguration,
          errorOnMissing = false)
      }
      .getOrElse(getRequiredParam(inputDataDirectories).toSeq)
      .map(_.toString)

    logger.info(s"inputRecordsPath:\n${inputPathsWithRanges.mkString("\n")}")

    val numExecutors = sc.getExecutorStorageStatus.length
    val minPartitions = if (sc.getConf.contains("spark.default.parallelism")) {
      sc.defaultParallelism
    } else {
      numExecutors * PARALLELISM_MODIFIER
    }
    val records = AvroUtils.readAvroFiles(sc, inputPathsWithRanges, minPartitions)
    // numExecutors * 5 is too much for the distinct operation when the data is huge; use numExecutors instead.
    val nameAndTermFeatureSetContainer = AvroUtils.readNameAndTermFeatureSetContainerFromGenericRecords(
      records,
      getRequiredParam(featureBagsKeys),
      numExecutors)

    nameAndTermFeatureSetContainer.saveAsTextFiles(getRequiredParam(rootOutputDirectory).toString, sc)
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
   * Entry point to the job.
   *
   * @param args The command line arguments for the job
   */
  def main(args: Array[String]): Unit = {

    // Parse and apply parameters
    val params: ParamMap = ScoptNameAndTermFeatureBagsParametersParser.parseFromCommandLine(args)
    params.toSeq.foreach(set)

    implicit val log = logger
    sc = SparkSessionConfiguration.asYarnClient(getOrDefault(applicationName), useKryo = true).sparkContext

    try {

      Timed("Total time in name and term feature bags driver")(run())

    } catch { case e: Exception =>

      log.error("Failure while running the driver", e)
      throw e

    } finally {

      sc.stop()
    }
  }
}
