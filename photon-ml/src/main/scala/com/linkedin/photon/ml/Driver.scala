package com.linkedin.photon.ml

import java.io.{OutputStreamWriter, PrintWriter}

import com.linkedin.photon.ml.diagnostics.hl.HosmerLemeshowDiagnostic
import com.linkedin.photon.ml.diagnostics.reporting.SectionPhysicalReport
import com.linkedin.photon.ml.diagnostics.reporting.html.HTMLRenderStrategy
import com.linkedin.photon.ml.diagnostics.reporting.reports.combined.{DiagnosticToPhysicalReportTransformer, DiagnosticReport}
import com.linkedin.photon.ml.diagnostics.reporting.reports.model.ModelDiagnosticReport
import com.linkedin.photon.ml.diagnostics.reporting.reports.system.SystemReport
import com.linkedin.photon.ml.supervised.TaskType
import TaskType._
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.io.{GLMSuite, LogWriter}
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.RegularizationContext
import com.linkedin.photon.ml.stat.{BasicStatistics, BasicStatisticalSummary}
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, BinaryClassifier}
import com.linkedin.photon.ml.supervised.model.{ModelTracker, GeneralizedLinearModel}
import com.linkedin.photon.ml.supervised.regression.{PoissonRegressionModel, LinearRegressionModel, Regression}
import com.linkedin.photon.ml.util.Utils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.{ListBuffer, ArrayBuffer}
import scala.xml.PrettyPrinter

/**
 * Driver for the MLEase core machine learning algorithms. The processing done in the driver include three main components:
 * <ul>
 * <li> Preprocess, which reads in the data in the raw form (e.g., Avro) and transform and index them into ML-Ease's
 * internal data structure </li>
 * <li> Train, which trains the model given the user's specified configurations and parameters
 * (through [[Params]]) </li>
 * <li> Validate, which validates the trained model using the validating data set, if provided, and select the best
 * model given the validating results </li>
 * </ul>
 * More detailed documentation can be found either through the comments and notations in the source code, or at
 * [[https://iwww.corp.linkedin.com/wiki/cf/display/ENGS/How+to+MLEase]].
 * @param params: The ML-Ease parameters [[Params]]], containing essential information
 *              for the underlying model training tasks.
 * @param sc: The Spark context.
 * @param logger: A temporary container to hold the driver's logs.
 * @author xazhang
 * @author yizhou
 * @author dpeng
 */
protected[ml] class Driver(protected val params: Params, protected val sc: SparkContext, protected val logger: LogWriter) {

  import Driver._

  protected val stageHistory: mutable.ArrayBuffer[DriverStage] = new ArrayBuffer[DriverStage]()
  private[this] val trainDataStorageLevel: StorageLevel = params.normalizationType match {
    case NormalizationType.NO_SCALING =>
      DEFAULT_STORAGE_LEVEL
    case _ =>
      // If the input data needs to be normalized, do not persist the raw input data into memory.
      LOW_PRIORITY_STORAGE_LEVEL
  }
  private[this] val suite: GLMSuite = new GLMSuite(params.fieldsNameType, params.addIntercept, params.constraintString)
  protected var stage: DriverStage = DriverStage.INIT
  private[this] var trainingData: RDD[LabeledPoint] = null

  private[this] val regularizationContext: RegularizationContext = new RegularizationContext(params.regularizationType, params.elasticNetAlpha)
  private[this] var featureNum: Int = -1
  private[this] var trainingDataNum: Int = -1
  private[this] var summaryOption: Option[BasicStatisticalSummary] = None
  private[this] var lambdaModelTuples: List[(Double, _ <: GeneralizedLinearModel)] = List.empty
  private[this] var lambdaModelTrackerTuplesOption: Option[List[(Double, ModelTracker)]] = None
  private[this] var diagnostic:DiagnosticReport = null

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
    logger.println(s"Input parameters: $params\n")

    val startTime = System.currentTimeMillis()

    preprocess()
    train()
    validate()

    /* Store all the learned models and log messages to their corresponding directories */
    val elapsed = (System.currentTimeMillis() - startTime) * 0.001
    logger.println(f"total time elapsed: $elapsed%.3f(s)")

    Utils.createHDFSDir(params.outputDir, sc.hadoopConfiguration)
    val finalModelsDir = new Path(params.outputDir, LEARNED_MODELS_TEXT).toString
    Utils.deleteHDFSDir(finalModelsDir, sc.hadoopConfiguration)
    suite.writeModelsInText(sc, lambdaModelTuples, finalModelsDir.toString)

    logger.println(s"Final models are written to: $finalModelsDir")
  }

  protected def preprocess(): Unit = {
    assertDriverStage(DriverStage.INIT)

    /* Preprocess the data for the following model training and validating procedure using the chosen suite */
    val startTimeForPreprocessing = System.currentTimeMillis()

    trainingData = suite.readLabeledPointsFromAvro(sc, params.trainDir, params.minNumPartitions)
      .setName("training data").persist(trainDataStorageLevel)

    featureNum = trainingData.first().features.size
    trainingDataNum = trainingData.count().toInt

    /* Print out the basic statistics of the training data */
    logger.println(s"number of training data points: $trainingDataNum, number of features: $featureNum.")
    logger.println(s"Input RDD persisted in storage level $trainDataStorageLevel")

    val preprocessingTime = (System.currentTimeMillis() - startTimeForPreprocessing) * 0.001
    logger.println(f"preprocessing data finished, time elapsed: $preprocessingTime%.3f(s)")
    logger.flush()

    /* Summarize */
    for (summarizationDir <- params.summarizationOutputDirOpt) {
      summarizeFeatures(summarizationDir)
    }

    diagnostic = new DiagnosticReport(
      new SystemReport(suite.featureKeyToIdMap, params, summaryOption),
      new ListBuffer[ModelDiagnosticReport[GeneralizedLinearModel]]())

    updateStage(DriverStage.PREPROCESSED)
  }

  protected def summarizeFeatures(outputDir: String): Unit = {
    val beforeSummarization = System.currentTimeMillis()
    val summary = BasicStatistics.getBasicStatistics(trainingData)

    Utils.deleteHDFSDir(outputDir, sc.hadoopConfiguration)
    suite.writeBasicStatistics(sc, summary, outputDir)
    logger.println(s"Feature statistics written to $outputDir")

    val timeForSummarization = 1.0E-3 * (System.currentTimeMillis() - beforeSummarization)
    logger.println(f"Feature summary finishes. Time elapsed: $timeForSummarization%.3f")
    logger.flush()
    summaryOption = Some(summary)
  }

  protected def train(): Unit = {
    assertDriverStage(DriverStage.PREPROCESSED)

    /* Given the processed training data, starting to train a model using the chosen algorithm */
    val startTimeForTraining = System.currentTimeMillis()
    logger.println("model training started...")

    // The sole purpose of optimizationStateTrackersMapOption is used for logging. When we have better logging support, we should
    // remove stop returning optimizationStateTrackerMapOption
    val (_lambdaModelTuples, _lambdaModelTrackerTuplesOption) = ModelTraining.trainGeneralizedLinearModel(
      trainingData = trainingData,
      taskType = params.taskType,
      optimizerType = params.optimizerType,
      regularizationContext = regularizationContext,
      regularizationWeights = params.regularizationWeights,
      normalizationType = params.normalizationType,
      summaryOption = summaryOption,
      maxNumIter = params.maxNumIter,
      tolerance = params.tolerance,
      enableOptimizationStateTracker = params.enableOptimizationStateTracker,
      constraintMap = suite.constraintFeatureMap)
    lambdaModelTuples = _lambdaModelTuples
    lambdaModelTrackerTuplesOption = _lambdaModelTrackerTuplesOption

    val trainingTime = (System.currentTimeMillis() - startTimeForTraining) * 0.001
    logger.println(f"model training finished, time elapsed: $trainingTime%.3f(s)")

    lambdaModelTrackerTuplesOption.foreach { modelTrackersMap =>
      logger.println(s"optimization state tracker information:")
      modelTrackersMap.foreach { case (regularizationWeight, modelTracker) =>
        logger.println(s"model with regularization weight $regularizationWeight:")
        logger.println(s"${modelTracker.optimizationStateTracker}")
        logger.flush()
      }
    }
    trainingData.unpersist()

    updateStage(DriverStage.TRAINED)
  }

  protected def validate(): Unit = {
    assertDriverStage(DriverStage.TRAINED)
    if (params.validateDirOpt.isDefined) {
      val validateDir = params.validateDirOpt.get
      logger.println(s"\nRead validation data from $validateDir")

      // Read validation data after the training data are unpersisted.
      val validatingData = suite.readLabeledPointsFromAvro(sc, validateDir, params.minNumPartitions)
        .setName("validating data").persist(DEFAULT_STORAGE_LEVEL)


      var perModelMetrics:Map[Double, Map[String, Double]] = Map[Double, Map[String, Double]]()

      /* Validating the learned models using the validating data set */
      logger.println("\nStart to validate the learned models with validating data")
      logger.flush()
      val startTimeForValidating = System.currentTimeMillis()
      if (params.validatePerIteration) {
        // Calculate metrics for all (models, iterations)
        lambdaModelTrackerTuplesOption.foreach { weightModelTrackerTuples =>
          weightModelTrackerTuples.foreach { case (lambda, modelTracker) =>

            val msg = modelTracker.models.map(model => Evaluation.evaluate(model, validatingData)).zipWithIndex.map( x => {
              val (m, idx) = x
              m.keys.toSeq.sorted.map( y => {
                f"Iteration: [$idx%6d] Metric: [$y] value: ${m.get(y).get}"
              }).mkString("\n") }).mkString("\n")

            logger.println(s"Model with lambda = $lambda:\n$msg")
          }
        }
      } else {
        // Calculate metrics for all models
        lambdaModelTuples.foreach { case (lambda:Double, model: GeneralizedLinearModel) =>
          val metrics = Evaluation.evaluate(model, validatingData)
          perModelMetrics += (lambda -> metrics)
          val msg = metrics.keys.toSeq.sorted.map( y => {
            f"    Metric: [$y] value: ${metrics.get(y).get}"
          }).mkString("\n")
          logger.println(s"Model with lambda = $lambda:\nmsg")
        }
      }

      if (null != diagnostic) {
        val hlDiagnostic = new HosmerLemeshowDiagnostic()

        diagnostic.modelReports ++= lambdaModelTuples.map(x => {
          val (lambda, model) = x
          val metrics = perModelMetrics.getOrElse(lambda, Map.empty)
          model match {
            case lm: LogisticRegressionModel =>
              new ModelDiagnosticReport[GeneralizedLinearModel](
                model,
                lambda,
                lm.getClass.getName,
                suite.featureKeyToIdMap,
                metrics,
                summaryOption,
                Some(hlDiagnostic.diagnose(lm, validatingData, summaryOption)))

            case glm: GeneralizedLinearModel =>
              new ModelDiagnosticReport[GeneralizedLinearModel](
                model,
                lambda,
                glm.getClass.getName,
                suite.featureKeyToIdMap,
                metrics,
                summaryOption,
                None)
          }
        })

        writeDiagnostics(params.outputDir, REPORT_FILE, diagnostic)
      }

      logger.flush()

      val validatingTime = (System.currentTimeMillis() - startTimeForValidating) * 0.001
      logger.println(f"Model validating finished, time elapsed $validatingTime%.3f(s)")
      logger.flush()


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
      }

      logger.println(s"Regularization weight of the best model is: $bestModelWeight")
      val bestModelDir = new Path(params.outputDir, BEST_MODEL_TEXT).toString
      Utils.deleteHDFSDir(bestModelDir, sc.hadoopConfiguration)
      suite.writeModelsInText(sc, List((bestModelWeight, bestModel)), bestModelDir.toString)
      logger.println(s"The best model is written to: $bestModelDir")
      logger.flush()
      validatingData.unpersist()

      updateStage(DriverStage.VALIDATED)
    }
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
 * The container of the main function, where the input command line arguments are parsed into
 * [[Params]] via [[PhotonMLCmdLineParser]],
 * which in turn is to be consumed by the main Driver class.
 *
 * An example of running ML-Ease
 * through command line arguments can be found at [[https://iwww.corp.linkedin.com/wiki/cf/display/ENGS/How+to+MLEase]],
 * however, please note that our plan is to not have Driver or the command line as the user-facing interface for Photon,
 * instead, the template library should be.
 *
 * As a result, the Driver class will only serve as an internal developer's API.
 */
object Driver {
  val DEFAULT_STORAGE_LEVEL = StorageLevel.MEMORY_AND_DISK
  val LOW_PRIORITY_STORAGE_LEVEL = StorageLevel.DISK_ONLY
  val LEARNED_MODELS_TEXT = "learned-models-text"
  val BEST_MODEL_TEXT = "best-model-text"
  val REPORT_FILE = "diagnostic.html"
  val SUMMARY_CHAPTER = "Feature Summary"
  val MODEL_DIAGNOSTIC_CHAPTER = "Model Analysis"

  // Control pretty printer output
  val MAX_WIDTH = 120
  val INDENT = 2

  def main(args: Array[String]): Unit = {
    /* Parse the parameters from command line, should always be the 1st line in main*/
    val params = PhotonMLCmdLineParser.parseFromCommandLine(args)
    /* Configure the Spark application and initialize SparkContext, which is the entry point of a Spark application */
    val sc: SparkContext = SparkContextConfiguration.asYarnClient(new SparkConf(), params.jobName, params.kryo)

    // A temporary solution to save log into HDFS.
    val logger = new LogWriter(params.outputDir, sc)
    val job = new Driver(params, sc, logger)
    job.run()

    logger.close()
    sc.stop()
  }

  private def writeDiagnostics(outputDir: String, file: String, diagReport: DiagnosticReport): Unit = {
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