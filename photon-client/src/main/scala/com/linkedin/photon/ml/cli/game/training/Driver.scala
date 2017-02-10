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

import scala.collection.Map

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame
import org.slf4j.Logger

import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.cli.game.GAMEDriver
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.estimators.{GameEstimator, GameParams}
import com.linkedin.photon.ml.evaluation.Evaluator.EvaluationResults
import com.linkedin.photon.ml.io.ModelOutputMode
import com.linkedin.photon.ml.model.GAMEModel
import com.linkedin.photon.ml.util._

/**
 * The driver class, which provides the main entry point to GAME model training.
 */
final class Driver(val params: GameParams, val sparkContext: SparkContext, val logger: Logger)
  extends GAMEDriver(params, sparkContext, logger) {

  /**
   * Run the driver.
   */
  def run(): Unit = {

    val timer = new Timer

    // Process the output directory upfront and potentially fail the job early
    IOUtils.processOutputDir(params.outputDir, params.deleteOutputDirIfExists, sparkContext.hadoopConfiguration)

    // Load feature index maps
    timer.start()
    val featureIndexMapLoaders = prepareFeatureMaps()
    timer.stop()

    logger.info(s"Time elapsed after preparing feature maps: ${timer.durationSeconds} (s)\n")

    // Read training data
    val trainingData = readTrainingData(params.trainDirs, featureIndexMapLoaders)

    // Read validation data
    val validationData = params.validateDirsOpt.map { dirs =>
      timer.start()
      val data = readValidationData(dirs, featureIndexMapLoaders)
      timer.stop()

      logger.info("Time elapsed after validating data and evaluator preparation: " +
        s"${timer.durationSeconds} (s)\n")

      data
    }

    // Fit models
    val estimator = new GameEstimator(params, sparkContext, logger)
    val models = estimator.fit(trainingData, validationData)

    // Select best model
    timer.start()
    val bestModel = selectBestModel(models)
    timer.stop()

    logger.info(s"Time elapsed after selecting best model: ${timer.durationSeconds} (s)\n")

    // Write selected model
    if (params.modelOutputMode != ModelOutputMode.NONE) {
      timer.start()
      saveModelToHDFS(featureIndexMapLoaders, models, bestModel)
      timer.stop()

      logger.info(s"Time elapsed after saving game models to HDFS: ${timer.durationSeconds} (s)\n")
    }
  }

  /**
   * Reads the training dataset, handling specifics of input date ranges in the params.
   *
<<<<<<< HEAD:photon-client/src/main/scala/com/linkedin/photon/ml/cli/game/training/Driver.scala
   * @param trainDirs Path to the training data file(s)
   * @param featureIndexMapLoaders The feature index map loaders
   * @return The loaded data frame
=======
   * @param trainDirs path to the data file(s)
   * @param featureIndexMapLoaders the feature index map loaders
   * @return the loaded data frame
>>>>>>> Changes for scaling GAME scoring: add a new data structure ScoredGameDatum, implement replicated partitioned hash join:photon-ml/src/main/scala/com/linkedin/photon/ml/cli/game/training/Driver.scala
   */
  protected[training] def readTrainingData(
      trainDirs: Seq[String],
      featureIndexMapLoaders: Map[String, IndexMapLoader]): DataFrame = {

    // Get the training records paths
    val trainingRecordsPath = pathsForDateRange(trainDirs, params.trainDateRangeOpt, params.trainDateRangeDaysAgoOpt)
    logger.debug(s"Training records paths:\n${trainingRecordsPath.mkString("\n")}")

    // Determine the number of fixed effect partitions. Default to 0 if we have no fixed effects.
    val numFixedEffectPartitions = if (params.fixedEffectDataConfigurations.nonEmpty) {
      params.fixedEffectDataConfigurations.values.map(_.minNumPartitions).max
    } else {
      0
    }

    // Determine the number of random effect partitions. Default to 0 if we have no random effects.
    val numRandomEffectPartitions = if (params.randomEffectDataConfigurations.nonEmpty) {
      params.randomEffectDataConfigurations.values.map(_.numPartitions).max
    } else {
      0
    }

    val numPartitions = math.max(numFixedEffectPartitions, numRandomEffectPartitions)
    require(numPartitions > 0, "Invalid configuration: neither fixed effect nor random effect partitions specified.")

    // Read the data
    val dataReader = new AvroDataReader(sparkContext)
    dataReader.readMerged(
      trainingRecordsPath,
      featureIndexMapLoaders.toMap,
      params.featureShardIdToFeatureSectionKeysMap,
      numPartitions)
  }

  /**
   * Reads the validation dataset, handling specifics of input date ranges in the params.
   *
<<<<<<< HEAD:photon-client/src/main/scala/com/linkedin/photon/ml/cli/game/training/Driver.scala
   * @param validationDirs To the data file(s)
   * @param featureIndexMapLoaders The feature index map loaders
   * @return The loaded data frame
=======
   * @param validationDirs path to the data file(s)
   * @param featureIndexMapLoaders the feature index map loaders
   * @return the loaded data frame
>>>>>>> Changes for scaling GAME scoring: add a new data structure ScoredGameDatum, implement replicated partitioned hash join:photon-ml/src/main/scala/com/linkedin/photon/ml/cli/game/training/Driver.scala
   */
  protected[training] def readValidationData(
      validationDirs: Seq[String],
      featureIndexMapLoaders: Map[String, IndexMapLoader]): DataFrame = {

    // Get the validation records paths
    val validationRecordsPath = pathsForDateRange(
      validationDirs,
      params.validateDateRangeOpt,
      params.validateDateRangeDaysAgoOpt)

    logger.debug(s"Validation records paths:\n${validationRecordsPath.mkString("\n")}")

    // Read the data
    val dataReader = new AvroDataReader(sparkContext)
    dataReader.readMerged(
      validationRecordsPath,
      featureIndexMapLoaders.toMap,
      params.featureShardIdToFeatureSectionKeysMap,
      params.minPartitionsForValidation)
  }

  /**
   * Select best model according to validation evaluator.
   *
   * @param models The models to evaluate (single evaluator, on the validation data set)
   * @return The best model
   */
  protected[training] def selectBestModel(
      models: Seq[(GAMEModel, Option[EvaluationResults], String)]): Option[(GAMEModel, EvaluationResults, String)] = {

    val best = models
      .flatMap { case (model, evaluations, modelConfig) => evaluations.map((model, _, modelConfig)) }
      .reduceOption { (configModelEval1, configModelEval2) =>
        val (_, eval1, _) = configModelEval1
        val (_, eval2, _) = configModelEval2
        val (evaluator, score1) = eval1.head
        val (_, score2) = eval2.head

        if (evaluator.betterThan(score1, score2)) {
          configModelEval1
        } else {
          configModelEval2
        }
      }

    best match {
      case Some((model, eval, config)) =>
        logger.info(s"Evaluator ${eval.head._1.getEvaluatorName} selected model with the following config:\n" +
          s"$config\n" +
          s"Model summary:\n${model.toSummaryString}\n\n" +
          s"Evaluation result is : ${eval.head._2}")

      case _ =>
        logger.debug("No best model selection because no validation data was provided")
    }

    best
  }

  /**
   * Write the GAME models to HDFS.
   *
   * TODO: Deprecate model-spec then remove it in favor of model-metadata, but there are clients!
   *
<<<<<<< HEAD:photon-client/src/main/scala/com/linkedin/photon/ml/cli/game/training/Driver.scala
   * @param featureShardIdToFeatureMapLoader The shard ids
   * @param models All the models that were producing during training
   * @param bestModel The best model
=======
   * @param featureShardIdToFeatureMapLoader the shard ids
   * @param models all the models that were produced during training
   * @param bestModel the best model
>>>>>>> Changes for scaling GAME scoring: add a new data structure ScoredGameDatum, implement replicated partitioned hash join:photon-ml/src/main/scala/com/linkedin/photon/ml/cli/game/training/Driver.scala
   */
  protected[training] def saveModelToHDFS(
      featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader],
      models: Seq[(GAMEModel, Option[EvaluationResults], String)],
      bestModel: Option[(GAMEModel, EvaluationResults, String)]) {

    // Write the best model to HDFS
    bestModel match {

      case Some((model, _, modelConfig)) =>

        val modelOutputDir = new Path(params.outputDir, "best").toString
        Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)

        // TODO: deprecate this
        val modelSpecDir = new Path(modelOutputDir, "model-spec").toString
        IOUtils.writeStringsToHDFS(Iterator(modelConfig), modelSpecDir, hadoopConfiguration,
          forceOverwrite = false)

        ModelProcessingUtils.saveGameModelsToHDFS(model, featureShardIdToFeatureMapLoader, modelOutputDir,
          params, sparkContext)

        logger.info("Saved model to HDFS")

      case None =>
        logger.info("No model to save to HDFS")
    }

    // Write all models to HDFS
    // TODO: just output the best model once we have hyperparameter optimization
    if (params.modelOutputMode == ModelOutputMode.ALL) {
      models.foldLeft(0) { case (modelIndex, (model, _, modelConfig)) =>
        val modelOutputDir = new Path(params.outputDir, s"all/$modelIndex").toString

        // TODO: deprecate this
        val modelSpecDir = new Path(modelOutputDir, "model-spec").toString

        Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)
        IOUtils.writeStringsToHDFS(Iterator(modelConfig), modelSpecDir, hadoopConfiguration, forceOverwrite = false)
        ModelProcessingUtils.saveGameModelsToHDFS(
          model,
          featureShardIdToFeatureMapLoader,
          modelOutputDir,
          params,
          sparkContext)

        modelIndex + 1
      }
    }
  }
}

object Driver {

  val LOGS = "logs"

  /**
   * Main entry point.
   *
   * @param args
   */
  def main(args: Array[String]): Unit = {

    val timer = Timer.start()

    val params = GameParams.parseFromCommandLine(args)
    import params._

    val sc = SparkContextConfiguration.asYarnClient(applicationName, useKryo = true)

    val logsDir = new Path(outputDir, LOGS).toString
    val logger = new PhotonLogger(logsDir, sc)
    //TODO: This Photon log level should be made configurable
    logger.setLogLevel(PhotonLogger.LogLevelDebug)

    try {
      logger.debug(params.toString + "\n")

      val job = new Driver(params, sc, logger)
      job.run()

      timer.stop()
      logger.info(s"Overall time elapsed ${timer.durationMinutes} minutes")
    } catch {
      case e: Exception =>
        logger.error("Failure while running the driver", e)
        throw e
    } finally {
      logger.close()
      sc.stop()
    }
  }
}
