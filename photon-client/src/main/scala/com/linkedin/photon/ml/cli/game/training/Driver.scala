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
import org.apache.spark.sql.DataFrame
import org.slf4j.Logger

import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.Types.{FeatureShardId, SparkVector}
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.cli.game.GAMEDriver
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.estimators.{GameEstimator, GameParams}
import com.linkedin.photon.ml.evaluation.Evaluator.EvaluationResults
import com.linkedin.photon.ml.io.{GLMSuite, ModelOutputMode}
import com.linkedin.photon.ml.model.GAMEModel
import com.linkedin.photon.ml.normalization.{NormalizationContext, NormalizationType}
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.util.Implicits._
import com.linkedin.photon.ml.util.Utils._
import com.linkedin.photon.ml.util._

/**
 * The Driver class, which drives the training of GAME model.
 *
 * @note there is a separate Driver to drive the scoring of GAME models.
 */
final class Driver(val sc: SparkContext, val params: GameParams, implicit val logger: Logger)
  extends GAMEDriver(sc, params, logger) {

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

    Timed("clean output directories") {
      cleanOutputDirs()
    }
    val featureIndexMapLoaders = Timed("prepare features") {
      prepareFeatureMaps()
    }
    val trainingData = Timed("read training data") {
      readTrainingData(featureIndexMapLoaders)
    }
    val validationData = Timed("read validation data") {
      readValidationData(featureIndexMapLoaders)
    }
    val featureShardStats = Timed("calculate statistics for each feature shard") {
      calculateAndSaveFeatureShardStats(trainingData, featureIndexMapLoaders)
    }
    val normalizationContexts = Timed("prepare normalization contexts") {
      prepareNormalizationContexts(trainingData, featureIndexMapLoaders, featureShardStats)
    }
    val models = Timed("fit") {
      new GameEstimator(sc, params, logger).fit(trainingData, validationData, normalizationContexts)
    }
    val bestModel = Timed("select best model") {
      selectBestModel(models)
    }
    Timed("save model") {
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
   * @param featureIndexMapLoaders The feature index map loaders
   * @return The loaded data frame
   */
  protected[training] def readTrainingData(featureIndexMapLoaders: Map[String, IndexMapLoader]): DataFrame = {

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

    new AvroDataReader(sc).readMerged(
      trainingRecordsPath,
      featureIndexMapLoaders,
      params.featureShardIdToFeatureSectionKeysMap,
      numPartitions)
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
          (featureShardId, BasicStatisticalSummary(data.select(featureShardId).map(_.getAs[SparkVector](0))))
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
              IOUtils.writeBasicStatistics(sc, featureShardStats, outputDir, indexMap)
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
          val intercept = featureIndexMapLoaders(featureShardId).indexMapForDriver().get(GLMSuite.INTERCEPT_NAME_TERM)
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
      models: Seq[(GAMEModel, Option[EvaluationResults], String)]): Option[(GAMEModel, EvaluationResults, String)] =

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
            s"Model summary:\n${model.toSummaryString}\n\n" +
            s"Evaluation result is : ${eval.head._2}")
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
      models: Seq[(GAMEModel, Option[EvaluationResults], String)],
      bestModel: Option[(GAMEModel, EvaluationResults, String)]): Unit =

    if (params.modelOutputMode != ModelOutputMode.NONE) {

      // Write the best model to HDFS
      bestModel match {

        case Some((model, _, modelConfig)) =>

          val modelOutputDir = new Path(params.outputDir, "best").toString
          Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)

          val modelSpecDir = new Path(modelOutputDir, "model-spec").toString
          IOUtils.writeStringsToHDFS(Iterator(modelConfig), modelSpecDir, hadoopConfiguration,
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
            IOUtils.writeStringsToHDFS(Iterator(modelConfig), modelSpecDir, hadoopConfiguration, forceOverwrite = false)
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
        //TODO: This Photon log level should be made configurable
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
