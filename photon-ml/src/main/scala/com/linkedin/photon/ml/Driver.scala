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
package com.linkedin.photon.ml

import java.io.{IOException, OutputStreamWriter, PrintWriter}

import com.linkedin.photon.ml.data.{DataValidators, LabeledPoint}
import com.linkedin.photon.ml.diagnostics.DiagnosticMode
import com.linkedin.photon.ml.diagnostics.bootstrap.{BootstrapReport, BootstrapTrainingDiagnostic}
import com.linkedin.photon.ml.diagnostics.featureimportance.{ExpectedMagnitudeFeatureImportanceDiagnostic, FeatureImportanceReport, VarianceFeatureImportanceDiagnostic}
import com.linkedin.photon.ml.diagnostics.fitting.{FittingDiagnostic, FittingReport}
import com.linkedin.photon.ml.diagnostics.hl.{HosmerLemeshowDiagnostic, HosmerLemeshowReport}
import com.linkedin.photon.ml.diagnostics.independence.{PredictionErrorIndependenceDiagnostic, PredictionErrorIndependenceReport}
import com.linkedin.photon.ml.diagnostics.reporting.html.HTMLRenderStrategy
import com.linkedin.photon.ml.diagnostics.reporting.reports.combined.{DiagnosticReport, DiagnosticToPhysicalReportTransformer}
import com.linkedin.photon.ml.diagnostics.reporting.reports.model.ModelDiagnosticReport
import com.linkedin.photon.ml.diagnostics.reporting.reports.system.SystemReport
import com.linkedin.photon.ml.io.{GLMSuite, InputDataFormat, InputFormatFactory}
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext, NormalizationType}
import com.linkedin.photon.ml.optimization.RegularizationContext
import com.linkedin.photon.ml.stat.{BasicStatisticalSummary, BasicStatistics}
import com.linkedin.photon.ml.supervised.TaskType._
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.util.{IOUtils, PalDBIndexMapLoader, PhotonLogger, Utils}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.Logger

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.xml.PrettyPrinter

/**
  * Driver for the Photon-ML core machine learning algorithms. The processing done in the driver include three main
  * components:
  * <ul>
  * <li> Preprocess, which reads in the data in the raw form (e.g., Avro) and transform and index them into Photon-ML's
  * internal data structure </li>
  * <li> Train, which trains the model given the user's specified configurations and parameters
  * (through [[Params]]) </li>
  * <li> Validate, which validates the trained model using the validating data set, if provided, and select the best
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
  protected val seed: Long = System.nanoTime) {

  import com.linkedin.photon.ml.Driver._

  private[this] var inputDataFormat: InputDataFormat = null

  protected val stageHistory: mutable.ArrayBuffer[DriverStage] = new ArrayBuffer[DriverStage]()
  private[this] val trainDataStorageLevel: StorageLevel = DEFAULT_STORAGE_LEVEL
  protected var stage: DriverStage = DriverStage.INIT
  private[this] var trainingData: RDD[LabeledPoint] = null
  private[this] var validatingData: RDD[LabeledPoint] = null

  private[this] val regularizationContext: RegularizationContext =
    new RegularizationContext(params.regularizationType, params.elasticNetAlpha)
  private[this] var featureNum: Int = -1
  private[this] var trainingDataNum: Int = -1

  private[this] var summaryOption: Option[BasicStatisticalSummary] = None
  private[this] var normalizationContext: NormalizationContext = NoNormalization

  private[this] var lambdaModelTuples: List[(Double, _ <: GeneralizedLinearModel)] = List.empty
  private[this] var lambdaModelTrackerTuplesOption: Option[List[(Double, ModelTracker)]] = None
  private[this] var diagnostic: DiagnosticReport = null
  private[this] var perModelMetrics: Map[Double, Map[String, Double]] = Map[Double, Map[String, Double]]()

  def numFeatures(): Int = {
    assertEqualOrAfterDriverStage(DriverStage.PREPROCESSED)
    featureNum
  }

  def numTrainingData(): Int = {
    assertEqualOrAfterDriverStage(DriverStage.PREPROCESSED)
    trainingDataNum
  }

  protected def assertEqualOrAfterDriverStage(afterStage: DriverStage): Unit = {
    if (stage < afterStage) {
      throw new RuntimeException("Expecting driver stage to be equal or after " + afterStage + " but actually it is " +
        stage)
    }
  }

  def run(): Unit = {
    logger.info(s"Input parameters: \n$params\n")

    val startTime = System.currentTimeMillis()

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

    if (params.validateDirOpt.isDefined) {
      assertDriverStage(DriverStage.TRAINED)
      validate()
      updateStage(DriverStage.VALIDATED)
    }

    if (params.diagnosticMode != DiagnosticMode.NONE) {
      if (params.validateDirOpt.isDefined) {
        assertDriverStage(DriverStage.VALIDATED)
      } else {
        assertDriverStage(DriverStage.TRAINED)
      }
      diagnose()
      updateStage(DriverStage.DIAGNOSED)
    }

    // Unpersist the training and validation data
    trainingData.unpersist()
    if (params.validateDirOpt.isDefined) {
      validatingData.unpersist()
    }

    /* Store all the learned models and log messages to their corresponding directories */
    val elapsed = (System.currentTimeMillis() - startTime) * 0.001
    logger.info(f"total time elapsed: $elapsed%.3f(s)")

    Utils.createHDFSDir(params.outputDir, sc.hadoopConfiguration)
    val finalModelsDir = new Path(params.outputDir, LEARNED_MODELS_TEXT).toString
    IOUtils.writeModelsInText(sc, lambdaModelTuples, finalModelsDir.toString, inputDataFormat.indexMapLoader())

    logger.info(s"Final models are written to: $finalModelsDir")
  }

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
      "No training data found. Ensure that training data exists and feature vectors are not empty.")

    featureNum = trainingData.first().features.size

    // Print out the basic statistics of the training data
    logger.info(s"Number of training data points: $trainingDataNum, " +
      s"total number of unique features found in training data including intercept (if any): $featureNum.")
    logger.info(s"Input RDD persisted in storage level $trainDataStorageLevel")

    if (! DataValidators.sanityCheckData(trainingData, params.taskType, params.dataValidationType)) {
      throw new IllegalArgumentException("Training data has issues")
    }
  }

  protected def prepareValidatingData(validateDir: String): Unit = {
    logger.info(s"\nRead validation data from $validateDir")

    // Read validation data after the training data are unpersisted.
    validatingData = inputDataFormat
      .loadLabeledPoints(sc, validateDir, params.selectedFeaturesFile, params.minNumPartitions)
      .persist(trainDataStorageLevel).setName("validating data")
    if (! DataValidators.sanityCheckData(validatingData, params.taskType, params.dataValidationType)) {
      throw new IllegalArgumentException("Validation data has issues")
    }
  }

  protected def summarizeFeatures(outputDir: Option[String]): BasicStatisticalSummary = {
    val beforeSummarization = System.currentTimeMillis()
    val summary = BasicStatistics.getBasicStatistics(trainingData)

    outputDir.foreach { dir =>
      IOUtils.writeBasicStatistics(sc,
        summary,
        dir,
        inputDataFormat.indexMapLoader().indexMapForDriver())
      logger.info(s"Feature statistics written to $outputDir")
    }

    val timeForSummarization = 1.0E-3 * (System.currentTimeMillis() - beforeSummarization)
    logger.info(f"Feature summary finishes. Time elapsed: $timeForSummarization%.3f")

    summary
  }

  @throws(classOf[IOException])
  @throws(classOf[IllegalArgumentException])
  protected def preprocess(): Unit = {

    /* Preprocess the data for the following model training and validating procedure using the chosen suite */
    val startTimeForPreprocessing = System.currentTimeMillis()

    inputDataFormat = InputFormatFactory.createInputFormat(sc, params)

    prepareTrainingData()

    params.validateDirOpt.foreach { validateDir =>
      prepareValidatingData(validateDir)
    }

    /* Summarize */
    if (params.summarizationOutputDirOpt.isDefined || params.normalizationType != NormalizationType.NONE) {
      val summary = summarizeFeatures(params.summarizationOutputDirOpt)
      summaryOption = Some(summary)
      normalizationContext = NormalizationContext(
        params.normalizationType,
        summary,
        inputDataFormat.indexMapLoader().indexMapForDriver().get(GLMSuite.INTERCEPT_NAME_TERM))
    }

    val preprocessingTime = (System.currentTimeMillis() - startTimeForPreprocessing) * 0.001
    logger.info(f"preprocessing data finished, time elapsed: $preprocessingTime%.3f(s)")

  }

  protected def train(): Unit = {

    /* Given the processed training data, starting to train a model using the chosen algorithm */
    val startTimeForTraining = System.currentTimeMillis()
    logger.info("model training started...")

    // The sole purpose of optimizationStateTrackersMapOption is used for logging. When we have better logging support,
    // we should remove stop returning optimizationStateTrackerMapOption
    val (_lambdaModelTuples, _lambdaModelTrackerTuplesOption) = ModelTraining.trainGeneralizedLinearModel(
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
      treeAggregateDepth = params.treeAggregateDepth)
    lambdaModelTuples = _lambdaModelTuples
    lambdaModelTrackerTuplesOption = _lambdaModelTrackerTuplesOption

    val trainingTime = (System.currentTimeMillis() - startTimeForTraining) * 0.001
    logger.info(f"model training finished, time elapsed: $trainingTime%.3f(s)")

    lambdaModelTrackerTuplesOption.foreach { modelTrackersMap =>
      logger.info(s"optimization state tracker information:")

      modelTrackersMap.foreach { case (regularizationWeight, modelTracker) =>
        logger.info(s"model with regularization weight $regularizationWeight: " +
                    s"${modelTracker.optimizationStateTrackerString}")
      }
    }
  }

  protected def computeAndLogModelMetrics(): Unit = {
    if (params.validatePerIteration) {
      // Calculate metrics for all (models, iterations)
      lambdaModelTrackerTuplesOption.foreach { weightModelTrackerTuples =>
        weightModelTrackerTuples.foreach { case (lambda, modelTracker) =>
          val msg = modelTracker
            .models
            .map(Evaluation.evaluate(_, validatingData))
            .zipWithIndex
            .map { x =>
              val (m, idx) = x

              m.keys
                .toSeq
                .sorted
                .map(y => f"Iteration: [$idx%6d] Metric: [$y] value: ${m.get(y).get}")
                .mkString("\n")
            }
            .mkString("\n")

          logger.info(s"Model with lambda = $lambda:\n$msg")
        }
      }
    } else {
      // Calculate metrics for all models
      lambdaModelTuples.foreach { case (lambda: Double, model: GeneralizedLinearModel) =>
        val metrics = Evaluation.evaluate(model, validatingData)
        val msg = metrics
          .keys
          .toSeq
          .sorted
          .map(y => f"    Metric: [$y] value: ${metrics.get(y).get}")
          .mkString("\n")

        logger.info(s"Model with lambda = $lambda:\n$msg")

        perModelMetrics += (lambda -> metrics)
      }
    }
  }

  protected def modelSelection(): Unit = {
    // TODO: we potentially have an excessive memory usage issue at this step, 2M feature dataset with fail at here
    // due to OOM

    /* Select the best model using the validating data set and stores the best model as text file */
    val (bestModelWeight, bestModel: GeneralizedLinearModel) = params.taskType match {
      case LINEAR_REGRESSION =>
        val models = lambdaModelTuples.map(x => (x._1, x._2.asInstanceOf[LinearRegressionModel]))
        ModelSelection.selectBestLinearRegressionModel(models, validatingData)
      case POISSON_REGRESSION =>
        val models = lambdaModelTuples.map(x => (x._1, x._2.asInstanceOf[PoissonRegressionModel]))
        ModelSelection.selectBestPoissonRegressionModel(models, validatingData)
      case LOGISTIC_REGRESSION =>
        val models = lambdaModelTuples.map(x => (x._1, x._2.asInstanceOf[LogisticRegressionModel]))
        ModelSelection.selectBestLinearClassifier(models, validatingData)
      case SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        val models = lambdaModelTuples.map(x => (x._1, x._2.asInstanceOf[SmoothedHingeLossLinearSVMModel]))
        ModelSelection.selectBestLinearClassifier(models, validatingData)
    }

    logger.info(s"Regularization weight of the best model is: $bestModelWeight")
    val bestModelDir = new Path(params.outputDir, BEST_MODEL_TEXT).toString
    IOUtils.writeModelsInText(sc,
      List((bestModelWeight, bestModel)),
      bestModelDir.toString,
      inputDataFormat.indexMapLoader()
    )
    logger.info(s"The best model is written to: $bestModelDir")
  }

  protected def validate(): Unit = {
    /* Validating the learned models using the validating data set */
    logger.info("\nStart to validate the learned models with validating data")
    val startTimeForValidating = System.currentTimeMillis()
    computeAndLogModelMetrics()
    modelSelection()

    val validatingTime = (System.currentTimeMillis() - startTimeForValidating) * 0.001
    logger.info(f"Model validating finished, time elapsed $validatingTime%.3f(s)")
  }

  protected def initializeDiagnosticReport(): Unit = {
    diagnostic = new DiagnosticReport(
      new SystemReport(inputDataFormat.indexMapLoader().indexMapForDriver(), params, summaryOption),
      new ListBuffer[ModelDiagnosticReport[GeneralizedLinearModel]]())
  }

  protected def trainFunc(x: RDD[LabeledPoint], y: Map[Double, GeneralizedLinearModel])
  : List[(Double, _ <: GeneralizedLinearModel)] = {

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
      treeAggregateDepth = params.treeAggregateDepth)._1
  }

  protected def trainDiagnostic(): (Map[Double, FittingReport], Map[Double, BootstrapReport]) = {
    val trainDiagnosticStart = System.currentTimeMillis
    logger.info(s"Starting training diagnostics")
    val lambdaModelMap = lambdaModelTuples.toMap[Double, GeneralizedLinearModel]
    val lambdaFitMap = new FittingDiagnostic().diagnose(trainFunc, lambdaModelMap, trainingData, summaryOption, seed)
    val lambdaBootstrapMap = new BootstrapTrainingDiagnostic(inputDataFormat.indexMapLoader().indexMapForDriver())
        .diagnose(trainFunc, lambdaModelMap, trainingData, summaryOption)
    val trainDiagnosticTime = (System.currentTimeMillis - trainDiagnosticStart) / 1000.0
    logger.info(f"Training diagnostic time elapsed: $trainDiagnosticTime%.03f(s)")
    (lambdaFitMap, lambdaBootstrapMap)
  }

  protected def modelDiagnosticWithValidateData()
  : Map[Double, (FeatureImportanceReport, FeatureImportanceReport, PredictionErrorIndependenceReport)]= {

    val modelDiagnosticStart = System.currentTimeMillis
    logger.info(s"Starting model diagnostics")
    val meanImportanceDiagnostic = new ExpectedMagnitudeFeatureImportanceDiagnostic(
      inputDataFormat.indexMapLoader().indexMapForDriver())

    val varImportanceDiagnostic = new VarianceFeatureImportanceDiagnostic(
      inputDataFormat.indexMapLoader().indexMapForDriver())

    val predictionErrorDiagnostic = new PredictionErrorIndependenceDiagnostic()
    val lambdaMeanVarImportancePredictionErrorMap =
      lambdaModelTuples.map(x => (x._1,
        (meanImportanceDiagnostic.diagnose(x._2, validatingData, summaryOption),
        varImportanceDiagnostic.diagnose(x._2, validatingData, summaryOption),
        predictionErrorDiagnostic.diagnose(x._2, validatingData, summaryOption)))
      ).toMap

    val modelDiagnosticTime = (System.currentTimeMillis - modelDiagnosticStart) / 1000.0
    logger.info(f"Model diagnostic time elapsed: $modelDiagnosticTime%.03f(s)")
    lambdaMeanVarImportancePredictionErrorMap
  }

  protected def validateDiagnostic(): (
      Map[Double, (FeatureImportanceReport, FeatureImportanceReport, PredictionErrorIndependenceReport)],
      Map[Double, Option[HosmerLemeshowReport]]) = {

    val validateDiagnosticStart = System.currentTimeMillis
    logger.info(s"Starting validate diagnostics")

    val lambdaMeanVarImportancePredictionErrorMap = modelDiagnosticWithValidateData()

    val hlDiagnostic = new HosmerLemeshowDiagnostic()
    val lambdaHLReport = lambdaModelTuples.map { case Pair(lambda, model) =>
      val hlReport = model match {
        case lm: LogisticRegressionModel => Some(hlDiagnostic.diagnose(lm, validatingData, summaryOption))
        case _ => None
      }
      (lambda, hlReport)
    }.toMap
    val validateDiagnosticTime = (System.currentTimeMillis - validateDiagnosticStart) / 1000.0
    logger.info(f"Validating diagnostic time elapsed: $validateDiagnosticTime%.03f(s)")
    (lambdaMeanVarImportancePredictionErrorMap, lambdaHLReport)
  }

  protected def reportDiagnosticResult(
      lambdaMeanVarImportancePredictionErrorMap:
        Map[Double, (FeatureImportanceReport, FeatureImportanceReport, PredictionErrorIndependenceReport)],
      lambdaFitMap: Map[Double, FittingReport],
      lambdaBootstrapMap: Map[Double, BootstrapReport],
      lambdaHLReport: Map[Double, Option[HosmerLemeshowReport]]): Unit = {

    val lambdaMeanImpactFeatureImportance = lambdaMeanVarImportancePredictionErrorMap.mapValues(_._1)
    val lambdaVarianceImpactFeatureImportance = lambdaMeanVarImportancePredictionErrorMap.mapValues(_._2)
    val lambdaPredictionErrorIndependenceReport = lambdaMeanVarImportancePredictionErrorMap.mapValues(_._3)

    diagnostic.modelReports ++= lambdaModelTuples.map { case (lambda, model) =>
      ModelDiagnosticReport[GeneralizedLinearModel](
        model,
        lambda,
        s"${model.getClass.getName} @ lambda = $lambda",
        inputDataFormat.indexMapLoader().indexMapForDriver(),
        metrics = perModelMetrics.getOrElse(lambda, Map.empty),
        summaryOption,
        predictionErrorIndependence = lambdaPredictionErrorIndependenceReport.get(lambda),
        hosmerLemeshow = lambdaHLReport.getOrElse(lambda, None),
        meanImpactFeatureImportance = lambdaMeanImpactFeatureImportance.get(lambda),
        varianceImpactFeatureImportance = lambdaVarianceImpactFeatureImportance.get(lambda),
        fitReport = lambdaFitMap.get(lambda),
        bootstrapReport = lambdaBootstrapMap.get(lambda))
    }
  }

  protected def diagnose(): Unit = {

    val startTimeForDiagnostics = System.currentTimeMillis
    initializeDiagnosticReport()

    val (lambdaFitMap, lambdaBootstrapMap) = params.diagnosticMode match {
      case DiagnosticMode.TRAIN | DiagnosticMode.ALL => trainDiagnostic()
      case _ => (Map[Double, FittingReport](), Map[Double, BootstrapReport]())
    }

    val (lambdaMeanVarImportancePredictionErrorMap, lambdaHLReport) =
      params.diagnosticMode match {
        case DiagnosticMode.VALIDATE | DiagnosticMode.ALL => validateDiagnostic()
        case _ => (Map[Double, (FeatureImportanceReport, FeatureImportanceReport, PredictionErrorIndependenceReport)](),
            Map[Double, Option[HosmerLemeshowReport]]())
      }

    reportDiagnosticResult(
      lambdaMeanVarImportancePredictionErrorMap,
      lambdaFitMap,
      lambdaBootstrapMap,
      lambdaHLReport)

    val totalDiagnosticTime = (System.currentTimeMillis - startTimeForDiagnostics) / 1000.0
    logger.info(f"Total diagnostic time: $totalDiagnosticTime%.03f (s)")
    logger.info("Writing diagnostics")
    writeDiagnostics(params.outputDir, REPORT_FILE, diagnostic)
  }

  protected def assertDriverStage(expectedStage: DriverStage): Unit = {
    if (stage < expectedStage) {
      throw new RuntimeException("Expecting driver stage to be " + expectedStage + " but actually it is before it." +
        " The actual stage is " + stage)
    } else if (stage > expectedStage) {
      throw new RuntimeException("Expecting driver stage to be " + expectedStage + " but actually it is after it." +
        " The actual stage is " + stage)
    }
  }

  // Used to track all historical stages this driver has gone through
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

  def main(args: Array[String]): Unit = {
    /* Parse the parameters from command line, should always be the 1st line in main*/
    val params = PhotonMLCmdLineParser.parseFromCommandLine(args)

    /* Configure the Spark application and initialize SparkContext, which is the entry point of a Spark application */
    val sc = SparkContextConfiguration.asYarnClient(new SparkConf(), params.jobName, params.kryo)

    // A temporary solution to save log into HDFS.
    val logPath = new Path(params.outputDir, "log-message.txt")
    val logger = new PhotonLogger(logPath, sc)
    //TODO: This Photon log level should be made configurable
    logger.setLogLevel(PhotonLogger.LogLevelDebug)

    try {
      val job = new Driver(params, sc, logger)
      job.run()

    } catch {
      case e: Exception =>
        logger.error("Failure while running the driver", e)
        throw e

    } finally {
      logger.close()
      sc.stop()
    }
  }

  protected def writeDiagnostics(outputDir: String, file: String, diagReport: DiagnosticReport): Unit = {
    val xform = new DiagnosticToPhysicalReportTransformer()
    val doc = xform.transform(diagReport)
    val rs = new HTMLRenderStrategy()
    val rendered = rs.locateRenderer(doc).render(doc)

    val hdfs = FileSystem.get(new Configuration())
    val fileStream = hdfs.create(new Path(s"$outputDir/$file"), true)
    val writer = new PrintWriter(new OutputStreamWriter(fileStream))

    try {
      val pp = new PrettyPrinter(MAX_WIDTH, INDENT)
      val buffer = new StringBuilder()
      pp.format(rendered, buffer)
      writer.println(buffer.toString)
      writer.flush()
      writer.close()
    } finally {
      fileStream.close()
    }
  }
}
