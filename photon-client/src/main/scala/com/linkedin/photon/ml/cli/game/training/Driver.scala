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
package com.linkedin.photon.ml.cli.game.training

import scala.util.{Failure, Success, Try}

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.Logger

import com.linkedin.photon.ml.Types.{FeatureShardId, SparkVector}
import com.linkedin.photon.ml.cli.game.GameDriver
import com.linkedin.photon.ml.data.avro.{AvroDataReader, ModelProcessingUtils}
import com.linkedin.photon.ml.estimators.{GameEstimator, GameParams}
import com.linkedin.photon.ml.evaluation.Evaluator.EvaluationResults
import com.linkedin.photon.ml.io.deprecated.ModelOutputMode
import com.linkedin.photon.ml.model.GameModel
import com.linkedin.photon.ml.normalization.{NormalizationContext, NormalizationType}
import com.linkedin.photon.ml.optimization.game.GameModelOptimizationConfiguration
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.util.Implicits._
import com.linkedin.photon.ml.util.Utils._
import com.linkedin.photon.ml.util._
import com.linkedin.photon.ml.{Constants, InputColumnsNames, SparkContextConfiguration}

/**
 * The Driver class, which drives the training of GAME model.
 *
 * @note there is a separate Driver to drive the scoring of GAME models.
 */
final class Driver(val sc: SparkContext, val params: GameParams, implicit val logger: Logger)
  extends GameDriver(sc, params, logger) {

  // These two types make the code easier to read, and are somewhat specific to the GAME Driver
  type FeatureShardStatistics = Iterable[(FeatureShardId, BasicStatisticalSummary)]
  type FeatureShardStatisticsOpt = Option[FeatureShardStatistics]
  type IndexMapLoaders = Map[FeatureShardId, IndexMapLoader]

  /**
   * Prepare the training data, fit models and select best model.
   * There is one model for each combination of fixed and random effect specified in the params.
   *
   * @note all intercept terms are turned ON by default in prepareFeatureMaps.
   */
  def run(): Unit = {

    Timed("Clean output directories") {
      cleanOutputDirs()
    }

    val featureIndexMapLoaders = Timed("Prepare features") {
      prepareFeatureMaps()
    }

    val trainingData = readAndCheck(readTrainingData(featureIndexMapLoaders), "training data").get
    val validationData = readAndCheck(readValidationData(featureIndexMapLoaders), "validation data")

    val featureShardStats = Timed("Calculate statistics for each feature shard") {
      calculateAndSaveFeatureShardStats(trainingData, featureIndexMapLoaders)
    }

    val normalizationContexts = Timed("Prepare normalization contexts") {
      prepareNormalizationContexts(trainingData, featureIndexMapLoaders, featureShardStats)
    }

    val models = Timed("Fit model") {
      new GameEstimator(sc, params, logger).fit(trainingData, validationData, normalizationContexts)
    }

    val bestModel = Timed("Select best model") {
      selectBestModel(models)
    }

    Timed("Save model") {
      saveModelToHDFS(featureIndexMapLoaders, models, bestModel)
    }
  }

  /**
   * Clean up the directories in which we are going to output the models.
   */
  protected[training] def cleanOutputDirs(): Unit = {

    val configuration = sc.hadoopConfiguration
    IOUtils.processOutputDir(params.outputDir, params.deleteOutputDirIfExists, configuration)
    params.summarizationOutputDirOpt
      .foreach(IOUtils.processOutputDir(_, params.deleteOutputDirIfExists, configuration))
  }

  /**
   * Reads the training dataset, handling specifics of input date ranges in the params.
   *
   * @note this returns an Option rather than a naked DataFrame to harmonize the signature with readValidationData.
   *       readTrainingData will most probably always return a "full" Option.
   *
   * @param featureIndexMapLoaders The feature index map loaders
   * @return An Option containing the loaded data frame
   */
  protected[training] def readTrainingData(featureIndexMapLoaders: Map[String, IndexMapLoader]): Option[DataFrame] = {

    val trainingRecordsPath =
      pathsForDateRange(params.trainDirs, params.trainDateRangeOpt, params.trainDateRangeDaysAgoOpt)

    logger.debug(s"Training records paths:\n${trainingRecordsPath.mkString("\n")}")

    val numFixedEffectPartitions = if (params.fixedEffectDataConfigurations.nonEmpty) {
      params.fixedEffectDataConfigurations.values.map(_.minNumPartitions).max
    } else {
      0
    }

    val numRandomEffectPartitions = if (params.randomEffectDataConfigurations.nonEmpty) {
      params.randomEffectDataConfigurations.values.map(_.numPartitions).max
    } else {
      0
    }

    val numPartitions = math.max(numFixedEffectPartitions, numRandomEffectPartitions)
    require(numPartitions > 0, "Invalid configuration: neither fixed effect nor random effect partitions specified.")

    Some(new AvroDataReader(sc).readMerged(
      trainingRecordsPath,
      featureIndexMapLoaders,
      params.featureShardIdToFeatureSectionKeysMap,
      numPartitions))
  }

  /**
   * Reads the validation dataset, handling specifics of input date ranges in the params.
   *
   * @param featureIndexMapLoaders The feature index map loaders
   * @return The loaded data frame
   */
  protected[training] def readValidationData(featureIndexMapLoaders: Map[String, IndexMapLoader]): Option[DataFrame] =

    params.validationDirsOpt.map {
      validationDirs =>

        val validationRecordsPath =
          pathsForDateRange(
            validationDirs,
            params.validationDateRangeOpt,
            params.validationDateRangeDaysAgoOpt)

        logger.debug(s"Validation records paths:\n${validationRecordsPath.mkString("\n")}")

        new AvroDataReader(sc)
          .readMerged(
            validationRecordsPath,
            featureIndexMapLoaders,
            params.featureShardIdToFeatureSectionKeysMap,
            params.minPartitionsForValidation)
    }

  /**
   * Check data, i.e. verify that it is admissible for training and/or validation.
   *
   * For now, we just check that the data contains at least a sample with non-zero weight). We throw out samples that
   * have a weight less than or equal to 0. If we don't find at least one sample with a strictly positive weight, we
   * throw an IllegalArgumentException, otherwise we return the filtered data set.
   *
   * @note this function has the side-effect of possibly reducing the size of the data set
   * @note this is somewhat expensive (it iterates over the whole data), so the user can control it via parameter
   *       checkData
   *
   * @param data The data to check
   * @return A data set where zero-weight samples have been removed, or an exception if we couldn't find at least one
   *         sample with a strictly positive weight
   */
  protected[training] def checkData(data: Option[DataFrame]): Option[DataFrame] = {

    val weightColumnName = params.inputColumnsNames(InputColumnsNames.WEIGHT)

    data.map { dataframe =>
      if (params.checkData) {
        val numBad = dataframe.filter(s"$weightColumnName <= 0.0").count()
        // This doesn't work with Spark 2.1.0
//        val numBad = dataframe
//          .map { row =>
//            if (row.getAs[Double](weightColumnName) <= 0.0) 1 else 0 }
//          .reduce(_ + _)
        require(numBad == 0, s"Found $numBad data points with weights <= 0. Please fix data set.")
      }
      dataframe
    }
  }

  /**
   * Helper to avoid writing the same code for training and validation data: read some data set, and then optionally
   * validate it, while timing each step separately.
   *
   * @param reader A function that will read a data set to an Option[DataFrame]
   * @param dataName The name of the data set to read and check ("training", "validation"...)
   * @return An Option[DataFrame] containing the data set if the checks are successul (an IllegalArgumentException is
   *         thrown otherwise)
   */
  protected[training] def readAndCheck(reader: => Option[DataFrame], dataName: String): Option[DataFrame] = {

    val tmp = Timed(s"Read $dataName") { reader }

    if (params.checkData)
      Timed(s"Check $dataName") { checkData(tmp) }
    else
      tmp
  }

  /**
   * Calculate basic statistics (same as spark-ml) on a DataFrame.
   *
   * @param data The data to compute statistics on
   * @param featureIndexMapLoaders The index map loaders to use to retrieve the feature shards
   * @return One BasicStatisticalSummary per feature shard
   */
  private def calculateStatistics(
      data: DataFrame,
      featureIndexMapLoaders: IndexMapLoaders): FeatureShardStatistics =

    featureIndexMapLoaders.keys
      .map {
        featureShardId =>
          // Calling rdd explicitly here to avoid a typed encoder lookup in Spark 2.1
          (featureShardId, BasicStatisticalSummary(data.select(featureShardId).rdd.map(_.getAs[SparkVector](0))))
      }

  /**
   * Compute basic statistics (same as spark-ml) of the training data for each feature shard.
   * At the same time, save those statistics to disk.
   *
   * @param trainingData The training data
   * @param featureIndexMapLoaders The index map loaders
   * @return Basic for each feature shard
   */
  protected[training] def calculateAndSaveFeatureShardStats(
      trainingData: DataFrame,
      featureIndexMapLoaders: IndexMapLoaders): FeatureShardStatisticsOpt =

    params.summarizationOutputDirOpt
      .map {
        (summarizationOutputDir: String) => {
          calculateStatistics(trainingData, featureIndexMapLoaders)
            .tap { case (featureShardId, featureShardStats) =>
              val outputDir = summarizationOutputDir + "/" + featureShardId
              val indexMap = featureIndexMapLoaders(featureShardId).indexMapForDriver()
              ModelProcessingUtils.writeBasicStatistics(sc, featureShardStats, outputDir, indexMap)
            }
        }
      }

  /**
   * Prepare normalization contexts, if the normalization options has been setup in the parameters.
   *
   * @param trainingData The training data
   * @param featureIndexMapLoaders The index map loaders
   * @return Normalization contexts for each featureShardId, or None if normalization is not needed
   */
  protected[training] def prepareNormalizationContexts(
      trainingData: DataFrame,
      featureIndexMapLoaders: IndexMapLoaders,
      statistics: FeatureShardStatisticsOpt): Option[Map[FeatureShardId, NormalizationContext]] =

    Filter(params.normalizationType != NormalizationType.NONE) {
      statistics
        .getOrElse(calculateStatistics(trainingData, featureIndexMapLoaders))
        .map { case (featureShardId, featureShardStats) =>
          val intercept = featureIndexMapLoaders(featureShardId).indexMapForDriver().get(Constants.INTERCEPT_KEY)
          (featureShardId, NormalizationContext(params.normalizationType, featureShardStats, intercept))
        }
        .toMap
    }

  /**
   * Select best model according to validation evaluator.
   *
   * @param models The models to evaluate (single evaluator, on the validation data set)
   * @return The best model
   */
  protected[training] def selectBestModel(
      models: Seq[(GameModel, Option[EvaluationResults], GameModelOptimizationConfiguration)])
    : Option[(GameModel, EvaluationResults, GameModelOptimizationConfiguration)] =

    models
      .flatMap { case (model, evaluations, modelConfig) => evaluations.map((model, _, modelConfig)) }
      .reduceOption { (configModelEval1, configModelEval2) =>
        val (eval1, eval2) = (configModelEval1._2, configModelEval2._2)
        val (evaluator, score1) = eval1.head
        val (_, score2) = eval2.head
        if (evaluator.betterThan(score1, score2)) configModelEval1 else configModelEval2
      }
      .tap {
        case (model, eval, config) =>
          logger.info(s"Evaluator ${eval.head._1.getEvaluatorName} selected model with the following config:\n" +
            s"$config\n" +
            s"Evaluation result is : ${eval.head._2}")

          // TODO: Computing model summary is slow, we should only do it if necessary
          if (logger.isDebugEnabled) {
            logger.debug(s"Model summary:\n${model.toSummaryString}\n")
          }
      }

  /**
   * Write the GAME models to HDFS.
   *
   * TODO: Deprecate model-spec then remove it in favor of model-metadata, but there are clients!
   * TODO: Should we perform model selection for NONE output mode? NONE output mode is used mostly as a debugging tool.
   *
   * @param featureShardIdToFeatureMapLoader The shard ids
   * @param models All the models that were producing during training
   * @param bestModel The best model
   */
  protected[training] def saveModelToHDFS(
      featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader],
      models: Seq[(GameModel, Option[EvaluationResults], GameModelOptimizationConfiguration)],
      bestModel: Option[(GameModel, EvaluationResults, GameModelOptimizationConfiguration)]): Unit =

    if (params.modelOutputMode != ModelOutputMode.NONE) {

      // Write the best model to HDFS
      bestModel match {

        case Some((model, _, modelConfig)) =>

          val modelOutputDir = new Path(params.outputDir, "best").toString
          Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)

          val modelSpecDir = new Path(modelOutputDir, "model-spec").toString
          IOUtils.writeStringsToHDFS(Iterator(modelConfig.toString()), modelSpecDir, hadoopConfiguration,
            forceOverwrite = false)

          ModelProcessingUtils.saveGameModelsToHDFS(model, featureShardIdToFeatureMapLoader, modelOutputDir,
            params, sc)

          logger.info("Saved model to HDFS")

        case None =>
          logger.info("No model to save to HDFS")
      }

      // Write all models to HDFS
      // TODO: just output the best model once we have hyperparameter optimization
      if (params.modelOutputMode == ModelOutputMode.ALL) {
        models.foldLeft(0) {
          case (modelIndex, (model, _, modelConfig)) =>

            val modelOutputDir = new Path(params.outputDir, s"all/$modelIndex").toString
            val modelSpecDir = new Path(modelOutputDir, "model-spec").toString

            Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)
            IOUtils.writeStringsToHDFS(Iterator(modelConfig.toString()), modelSpecDir, hadoopConfiguration,
              forceOverwrite = false)
            ModelProcessingUtils.saveGameModelsToHDFS(
              model,
              featureShardIdToFeatureMapLoader,
              modelOutputDir,
              params,
              sc)

            modelIndex + 1
        }
      }
    }
}

/**
 * This object is the main entry point for GAME's training Driver. There is another one for the scoring Driver.
 */
object Driver {

  private val LOGS = "logs"

  /**
   * Main entry point for GAME training driver.
   *
   * @param args The command line arguments
   */
  def main(args: Array[String]): Unit = {

    val tryParams = Try(GameParams.parseFromCommandLine(args)) // An exception can be thrown by parseFromCommandLine

    tryParams match {

      case Failure(e) =>
        println(s"Could not parse command line arguments to Game training driver correctly.\n" +
          s"Command line arguments (${args.length}) are:\n")
        args.foreach(println)
        throw e

      case Success(_) =>

        val params = tryParams.get
        val sc = SparkContextConfiguration.asYarnClient(params.applicationName, useKryo = true)
        val logsDir = new Path(params.outputDir, LOGS).toString
        implicit val logger = new PhotonLogger(logsDir, sc)
        // TODO: This Photon log level should be made configurable
        logger.setLogLevel(PhotonLogger.LogLevelInfo)
        logger.debug(params.toString + "\n")

        try {

          Timed("Total time in training Driver") {
            new Driver(sc, params, logger).run()
          }

        } catch {
          case e: Throwable =>
            logger.error("Failure while running the driver", e)
            throw e
        } finally {
          logger.close()
          sc.stop()
        }
    }
  }
}
