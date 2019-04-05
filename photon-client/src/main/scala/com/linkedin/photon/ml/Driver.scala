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
package com.linkedin.photon.ml

import java.io.IOException

import scala.collection.mutable

import org.apache.hadoop.fs.Path
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.Logger

import com.linkedin.photon.ml.data.avro.ModelProcessingUtils
import com.linkedin.photon.ml.data.{DataValidators, LabeledPoint}
import com.linkedin.photon.ml.evaluation.Evaluation
import com.linkedin.photon.ml.event._
import com.linkedin.photon.ml.io.deprecated.{InputDataFormat, InputFormatFactory}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext, NormalizationType}
import com.linkedin.photon.ml.optimization.{OptimizationStatesTracker, RegularizationContext}
import com.linkedin.photon.ml.stat.FeatureDataStatistics
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.util.{IOUtils, PhotonLogger, Utils}

/**
 * Driver for the Photon-ML core machine learning algorithms. The processing done in the driver include three main
 * components:
 * <ul>
 * <li> Preprocess, which reads in the data in the raw form (e.g., Avro) and transform and index them into Photon-ML's
 * internal data structure </li>
 * <li> Train, which trains the model given the user's specified configurations and parameters
 * (through [[Params]]) </li>
 * <li> Validate, which validates the trained model using the validating dataset, if provided, and select the best
 * model given the validating results </li>
 * </ul>
 * More detailed documentation can be found either through the comments and notations in the source code, or at
 * [[https://github.com/linkedin/photon-ml#example-scripts]].
 *
 * @param params: The Photon-ML parameters [[Params]]], containing essential information
 *              for the underlying model training tasks.
 * @param sc: The Spark context.
 * @param logger: A temporary container to hold the driver's logs.
 */
protected[ml] class Driver(
    protected val params: Params,
    protected val sc: SparkContext,
    protected val logger: Logger,
    protected val seed: Long = System.nanoTime)
  extends EventEmitter {

  import com.linkedin.photon.ml.Driver._

  //
  // Class members
  //

  private[this] val trainDataStorageLevel: StorageLevel = DEFAULT_STORAGE_LEVEL

  private[this] var inputDataFormat: InputDataFormat = null
  private[this] var trainingData: RDD[LabeledPoint] = null
  private[this] var validationData: RDD[LabeledPoint] = null

  private[this] val regularizationContext: RegularizationContext =
    RegularizationContext(params.regularizationType, params.elasticNetAlpha)
  private[this] var featureNum: Int = -1
  private[this] var trainingDataNum: Int = -1

  private[this] var summaryOption: Option[FeatureDataStatistics] = None
  private[this] var normalizationContext: NormalizationContext = NoNormalization()

  private[this] var lambdaModelAndTrackers: List[(Double, _ <: GeneralizedLinearModel, Option[OptimizationStatesTracker])] =
    List.empty

  protected var perModelMetrics: Map[Double, Evaluation.MetricsMap] = Map()

  protected val stageHistory: mutable.ArrayBuffer[DriverStage] = new mutable.ArrayBuffer[DriverStage]()
  protected var stage: DriverStage = DriverStage.INIT

  //
  // Class Initialization
  //

  params.eventListeners.foreach { eventListenerClass =>
    try {
      registerListener(Class.forName(eventListenerClass).getConstructor().newInstance().asInstanceOf[EventListener])

    } catch {
      case e: Exception =>
        throw new IllegalArgumentException(s"Error registering class $eventListenerClass as event listener", e)
    }
  }
  sendEvent(PhotonSetupEvent(logger, sc, params))

  //
  // Class Methods
  //

  /**
   *
   * @return
   */
  def numFeatures(): Int = {
    assertEqualOrAfterDriverStage(DriverStage.PREPROCESSED)
    featureNum
  }

  /**
   *
   * @return
   */
  def numTrainingData(): Int = {
    assertEqualOrAfterDriverStage(DriverStage.PREPROCESSED)
    trainingDataNum
  }

  /**
   *
   * @param afterStage
   */
  protected def assertEqualOrAfterDriverStage(afterStage: DriverStage): Unit = {
    if (stage < afterStage) {
      throw new RuntimeException("Expecting driver stage to be equal or after " + afterStage + " but actually it is " +
        stage)
    }
  }

  /**
   *
   */
  def run(): Unit = {
    logger.info(s"Input parameters: \n$params\n")

    val startTime = System.currentTimeMillis()
    sendEvent(TrainingStartEvent(startTime))

    // Process the output directory upfront and potentially fail the job early
    val configuration = sc.hadoopConfiguration
    IOUtils.processOutputDir(params.outputDir, params.deleteOutputDirsIfExist, configuration)
    params.summarizationOutputDirOpt.foreach(IOUtils.processOutputDir(_, params.deleteOutputDirsIfExist, configuration))

    assertDriverStage(DriverStage.INIT)
    preprocess()
    updateStage(DriverStage.PREPROCESSED)

    assertDriverStage(DriverStage.PREPROCESSED)
    train()
    updateStage(DriverStage.TRAINED)

    params.validateDirOpt match {
      case Some(_) =>
        assertDriverStage(DriverStage.TRAINED)
        validate()
        updateStage(DriverStage.VALIDATED)

      case _ =>
        lambdaModelAndTrackers.foreach { case (regWeight, _, trackerOpt) =>
            sendEvent(PhotonOptimizationLogEvent(regWeight, trackerOpt))
        }
    }

    // Unpersist the training and validation data
    trainingData.unpersist()
    if (params.validateDirOpt.isDefined) {
      validationData.unpersist()
    }

    // Store all the learned models and log messages to their corresponding directories
    val elapsed = (System.currentTimeMillis() - startTime) * 0.001
    logger.info(f"total time elapsed: $elapsed%.3f(s)")

    Utils.createHDFSDir(params.outputDir, sc.hadoopConfiguration)
    val finalModelsDir = new Path(params.outputDir, LEARNED_MODELS_TEXT)
    IOUtils.writeModelsInText(sc, lambdaModelAndTrackers, finalModelsDir.toString, inputDataFormat.indexMapLoader())

    logger.info(s"Final models are written to: $finalModelsDir")

    sendEvent(TrainingFinishEvent(System.currentTimeMillis()))
  }

  /**
   *
   */
  protected def prepareTrainingData(): Unit = {
    // Verify that selected features files exist
    params.selectedFeaturesFile.foreach(file => {
      val path = new Path(file)
      if (! path.getFileSystem(sc.hadoopConfiguration).exists(path)) {
        throw new IOException("Could not find [" + file + "]. Check that the file exists")
      }
    })

    trainingData = inputDataFormat
      .loadLabeledPoints(sc, params.trainDir, params.selectedFeaturesFile, params.minNumPartitions)
      .persist(trainDataStorageLevel)
      .setName("training data")

    trainingDataNum = trainingData.count().toInt
    require(trainingDataNum > 0,
      "No training data found. Ensure that training data exists and feature vectors are not empty, " +
        "and that at least one training sample has a weight > 0.")

    featureNum = trainingData.first().features.size

    // Print out the basic statistics of the training data
    logger.info(s"Number of training data points: $trainingDataNum, " +
      s"total number of unique features found in training data including intercept (if any): $featureNum.")
    logger.info(s"Input RDD persisted in storage level $trainDataStorageLevel")

    DataValidators.sanityCheckData(trainingData, params.taskType, params.dataValidationType)
  }

  /**
   *
   * @param validateDir
   */
  protected def prepareValidationData(validateDir: String): Unit = {
    logger.info(s"\nRead validation data from $validateDir")

    // Read validation data after the training data are unpersisted.
    validationData = inputDataFormat
      .loadLabeledPoints(sc, validateDir, params.selectedFeaturesFile, params.minNumPartitions)
      .persist(trainDataStorageLevel)
      .setName("validation data")

    require(trainingData.count() > 0,
      "No validation data found. Ensure that validation data exists and feature vectors are not empty, " +
        "and that at least one validation sample has a weight > 0.")

    DataValidators.sanityCheckData(validationData, params.taskType, params.dataValidationType)
  }

  /**
   *
   * @param outputDir
   * @return
   */
  protected def summarizeFeatures(outputDir: Option[Path]): FeatureDataStatistics = {

    val beforeSummarization = System.currentTimeMillis()
    val summary = FeatureDataStatistics(
      trainingData,
      inputDataFormat.indexMapLoader().indexMapForDriver().get(Constants.INTERCEPT_KEY))

    outputDir.foreach { dir =>
      ModelProcessingUtils.writeBasicStatistics(
        sc,
        summary,
        dir,
        inputDataFormat.indexMapLoader().indexMapForDriver())
      logger.info(s"Feature statistics written to $outputDir")
    }

    val timeForSummarization = 1.0E-3 * (System.currentTimeMillis() - beforeSummarization)
    logger.info(f"Feature summary finishes. Time elapsed: $timeForSummarization%.3f")

    summary
  }

  /**
   *
   * @throws java.io.IOException
   * @throws java.lang.IllegalArgumentException
   */
  @throws(classOf[IOException])
  @throws(classOf[IllegalArgumentException])
  protected def preprocess(): Unit = {
    // Preprocess the data for the following model training and validating procedure using the chosen suite
    val startTimeForPreprocessing = System.currentTimeMillis()

    inputDataFormat = InputFormatFactory.createInputFormat(sc, params)

    prepareTrainingData()

    params.validateDirOpt.foreach { validateDir =>
      prepareValidationData(validateDir)
    }

    // Summarize
    if (params.summarizationOutputDirOpt.isDefined || params.normalizationType != NormalizationType.NONE) {
      val summary = summarizeFeatures(params.summarizationOutputDirOpt)
      summaryOption = Some(summary)
      normalizationContext = NormalizationContext(params.normalizationType, summary)
    }

    val preprocessingTime = (System.currentTimeMillis() - startTimeForPreprocessing) * 0.001

    logger.info(f"preprocessing data finished, time elapsed: $preprocessingTime%.3f(s)")
  }

  /**
   *
   */
  protected def train(): Unit = {
    // Given the processed training data, starting to train a model using the chosen algorithm
    val startTimeForTraining = System.currentTimeMillis()
    logger.info("model training started...")

    // The sole purpose of optimizationStateTrackersMapOption is used for logging. When we have better logging support,
    // we should remove stop returning optimizationStateTrackerMapOption
    lambdaModelAndTrackers = ModelTraining.trainGeneralizedLinearModel(
      trainingData = trainingData,
      taskType = params.taskType,
      optimizerType = params.optimizerType,
      regularizationContext = regularizationContext,
      regularizationWeights = params.regularizationWeights,
      normalizationContext = normalizationContext,
      maxNumIter = params.maxNumIter,
      tolerance = params.tolerance,
      enableOptimizationStateTracker = params.enableOptimizationStateTracker,
      constraintMap = inputDataFormat.constraintFeatureMap(),
      treeAggregateDepth = params.treeAggregateDepth,
      useWarmStart = params.useWarmStart)

    val trainingTime = (System.currentTimeMillis() - startTimeForTraining) * 0.001
    logger.info(f"model training finished, time elapsed: $trainingTime%.3f(s)")

    if (params.enableOptimizationStateTracker) {
      logger.info(s"optimization state tracker information:")

      lambdaModelAndTrackers.foreach { abc =>
        logger.info(s"model with regularization weight ${abc._1}: ${abc._3.get.toSummaryString}")
      }
    }
  }

  /**
   *
   */
  protected def computeAndLogModelMetrics(): Unit =
    if (params.validatePerIteration) {
      lambdaModelAndTrackers.foreach { case (regularizationWeight: Double, glm: GeneralizedLinearModel, optimizationStatesTrackerOpt: Option[OptimizationStatesTracker]) =>
        require(
          optimizationStatesTrackerOpt.isDefined,
          s"Missing optimization state information for model with regularization weight $regularizationWeight")

        logger.info(s"Model with lambda = $regularizationWeight:")

        val tracker = optimizationStatesTrackerOpt.get
        val perIterationMetrics = tracker
          .getTrackedStates
          .map { optimizerState =>
            val model = glm.updateCoefficients(Coefficients(optimizerState.coefficients, None))
            val metrics = Evaluation.evaluate(model, validationData)

            metrics
              .keys
              .toSeq
              .sorted
              .foreach { metric =>
                logger.info(f"Iteration: [${optimizerState.iter}%6d] Metric: [$metric] value: ${metrics(metric)}")
              }

            (model, metrics)
          }

          val finalMetrics = perIterationMetrics.last._2

          perModelMetrics += (regularizationWeight -> finalMetrics)
          sendEvent(
            PhotonOptimizationLogEvent(
              regularizationWeight,
              optimizationStatesTrackerOpt,
              Some(perIterationMetrics),
              Some(finalMetrics)))
      }
    } else {
      lambdaModelAndTrackers.foreach { case (regularizationWeight: Double, glm: GeneralizedLinearModel, optimizationStatesTrackerOpt: Option[OptimizationStatesTracker]) =>
        logger.info(s"Model with lambda = $regularizationWeight:")

        val finalMetrics = Evaluation.evaluate(glm, validationData)

        finalMetrics
          .keys
          .toSeq
          .sorted
          .foreach { metric =>
            logger.info(f"Metric: [$metric] value: ${finalMetrics(metric)}")
          }

        perModelMetrics += (regularizationWeight -> finalMetrics)
        sendEvent(
          PhotonOptimizationLogEvent(regularizationWeight, optimizationStatesTrackerOpt, None, Some(finalMetrics)))
      }
    }

  /**
   *
   */
  protected def modelSelection(): Unit = {
    // TODO: we potentially have an excessive memory usage issue at this step, 2M feature dataset with fail at here
    // due to OOM

    /* Select the best model using the validating dataset and stores the best model as text file */
    val (bestModelWeight, bestModel: GeneralizedLinearModel) = params.taskType match {
      case TaskType.LINEAR_REGRESSION =>
        val models = lambdaModelAndTrackers.map(x => (x._1, x._2.asInstanceOf[LinearRegressionModel]))
        ModelSelection.selectBestLinearRegressionModel(models, perModelMetrics)
      case TaskType.POISSON_REGRESSION =>
        val models = lambdaModelAndTrackers.map(x => (x._1, x._2.asInstanceOf[PoissonRegressionModel]))
        ModelSelection.selectBestPoissonRegressionModel(models, perModelMetrics)
      case TaskType.LOGISTIC_REGRESSION =>
        val models = lambdaModelAndTrackers.map(x => (x._1, x._2.asInstanceOf[LogisticRegressionModel]))
        ModelSelection.selectBestLinearClassifier(models, perModelMetrics)
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        val models = lambdaModelAndTrackers.map(x => (x._1, x._2.asInstanceOf[SmoothedHingeLossLinearSVMModel]))
        ModelSelection.selectBestLinearClassifier(models, perModelMetrics)
    }
    val bestModelDir = new Path(params.outputDir, BEST_MODEL_TEXT).toString

    IOUtils.writeModelsInText(
      sc,
      List((bestModelWeight, bestModel, None)),
      bestModelDir.toString,
      inputDataFormat.indexMapLoader()
    )

    logger.info(s"Regularization weight of the best model is: $bestModelWeight")
    logger.info(s"The best model is written to: $bestModelDir")
  }

  /**
   *
   */
  protected def validate(): Unit = {
    /* Validating the learned models using the validating dataset */
    logger.info("Start to validate the learned models with validating data")
    val startTimeForValidating = System.currentTimeMillis()

    computeAndLogModelMetrics()
    modelSelection()

    val validatingTime = (System.currentTimeMillis() - startTimeForValidating) * 0.001
    logger.info(f"Model validating finished, time elapsed: $validatingTime%.3f(s)")
  }

  /**
   *
   * @param x
   * @param y
   * @return
   */
  protected def trainFunc(
      x: RDD[LabeledPoint],
      y: Map[Double, GeneralizedLinearModel]): List[(Double, _ <: GeneralizedLinearModel, Option[OptimizationStatesTracker])] =
    ModelTraining.trainGeneralizedLinearModel(
      trainingData = x,
      taskType = params.taskType,
      optimizerType = params.optimizerType,
      regularizationContext = regularizationContext,
      regularizationWeights = params.regularizationWeights,
      normalizationContext = normalizationContext,
      maxNumIter = params.maxNumIter,
      tolerance = params.tolerance,
      enableOptimizationStateTracker = params.enableOptimizationStateTracker,
      constraintMap = inputDataFormat.constraintFeatureMap(),
      warmStartModels = y,
      treeAggregateDepth = params.treeAggregateDepth,
      params.useWarmStart)

  /**
   *
   * @param expectedStage
   */
  protected def assertDriverStage(expectedStage: DriverStage): Unit = {
    if (stage < expectedStage) {
      throw new RuntimeException("Expecting driver stage to be " + expectedStage + " but actually it is before it." +
        " The actual stage is " + stage)
    } else if (stage > expectedStage) {
      throw new RuntimeException("Expecting driver stage to be " + expectedStage + " but actually it is after it." +
        " The actual stage is " + stage)
    }
  }

  /**
   * Used to track all historical stages this driver has gone through.
   *
   * @param nextStage
   */
  protected def updateStage(nextStage: DriverStage): Unit = {
    stageHistory += stage
    stage = nextStage
  }
}

/**
 * The container of the main function, where the input command line arguments are parsed into [[Params]] via
 * [[PhotonMLCmdLineParser]], which in turn is to be consumed by the main Driver class.
 *
 * An example of running Photon-ML through command line arguments can be found at
 * [[https://github.com/linkedin/photon-ml#example-scripts]].
 */
object Driver {
  val DEFAULT_STORAGE_LEVEL = StorageLevel.MEMORY_AND_DISK
  val LEARNED_MODELS_TEXT = "learned-models-text"
  val BEST_MODEL_TEXT = "best-model-text"
  val REPORT_FILE = "model-diagnostic.html"
  val SUMMARY_CHAPTER = "Feature Summary"
  val MODEL_DIAGNOSTIC_CHAPTER = "Model Analysis"

  // Control pretty printer output
  val MAX_WIDTH = 120
  val INDENT = 2

  /**
   *
   * @param args
   */
  def main(args: Array[String]): Unit = {
    // Parse the parameters from command line, should always be the 1st line in main
    val params = PhotonMLCmdLineParser.parseFromCommandLine(args)
    // Configure the Spark application and initialize SparkContext, which is the entry point of a Spark application
    val sc = SparkSessionConfiguration.asYarnClient(new SparkConf(), params.jobName, params.kryo).sparkContext
    // A temporary solution to save log into HDFS.
    val logPath = new Path(params.outputDir, "log-message.txt")
    val logger = new PhotonLogger(logPath, sc)

    // TODO: This Photon log level should be made configurable
    logger.setLogLevel(PhotonLogger.LogLevelDebug)

    try {
      val job = new Driver(params, sc, logger)
      job.run()
      job.clearListeners()

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
