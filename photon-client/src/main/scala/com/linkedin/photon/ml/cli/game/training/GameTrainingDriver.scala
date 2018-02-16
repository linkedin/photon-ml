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

import scala.math.log

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.linalg.{Vector => SparkMLVector}
import org.apache.spark.sql.DataFrame

import com.linkedin.photon.ml.HyperparameterTuningMode.HyperparameterTuningMode
import com.linkedin.photon.ml._
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types._
import com.linkedin.photon.ml.cli.game.GameDriver
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.{DataValidators, FixedEffectDataConfiguration, InputColumnsNames, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.data.avro.{AvroDataReader, ModelProcessingUtils}
import com.linkedin.photon.ml.estimators.GameEstimator.GameOptimizationConfiguration
import com.linkedin.photon.ml.estimators.{GameEstimator, GameEstimatorEvaluationFunction}
import com.linkedin.photon.ml.hyperparameter.search.{GaussianProcessSearch, RandomSearch}
import com.linkedin.photon.ml.index.IndexMapLoader
import com.linkedin.photon.ml.io.{CoordinateConfiguration, ModelOutputMode}
import com.linkedin.photon.ml.io.ModelOutputMode.ModelOutputMode
import com.linkedin.photon.ml.io.scopt.game.ScoptGameTrainingParametersParser
import com.linkedin.photon.ml.model.{DatumScoringModel, FixedEffectModel, RandomEffectModel}
import com.linkedin.photon.ml.normalization.NormalizationType.NormalizationType
import com.linkedin.photon.ml.normalization.{NormalizationContext, NormalizationType}
import com.linkedin.photon.ml.optimization.game.CoordinateOptimizationConfiguration
import com.linkedin.photon.ml.projector.IdentityProjection
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.util.Implicits._
import com.linkedin.photon.ml.util.Utils
import com.linkedin.photon.ml.util._

/**
 * This object is the entry point and driver for GAME training. There is a separate driver object for scoring.
 */
object GameTrainingDriver extends GameDriver {

  //
  // These types make the code easier to read, and are somewhat specific to the GAME Driver
  //

  type FeatureShardStatistics = Iterable[(FeatureShardId, BasicStatisticalSummary)]
  type FeatureShardStatisticsOpt = Option[FeatureShardStatistics]
  type IndexMapLoaders = Map[FeatureShardId, IndexMapLoader]

  //
  // Members
  //

  private val DEFAULT_APPLICATION_NAME = "GAME-Training"

  protected[training] val MODELS_DIR = "models"
  protected[training] val MODEL_SPEC_DIR = "model-spec"
  protected[training] val BEST_MODEL_DIR = "best"

  protected[training] var sc: SparkContext = _
  protected[training] implicit var logger: PhotonLogger = _

  //
  // Parameters
  //

  val trainingTask: Param[TaskType] = ParamUtils.createParam(
    "training task",
    "The type of training task to perform.",
    {taskType: TaskType => taskType != TaskType.NONE})

  val validationDataDirectories: Param[Set[Path]] = ParamUtils.createParam(
    "validation data directories",
    "Paths to directories containing validation data.",
    PhotonParamValidators.nonEmpty)

  val validationDataDateRange: Param[DateRange] = ParamUtils.createParam[DateRange](
    "validation data date range",
    "Inclusive date range for validation data. If specified, the validation directories are expected to be in the " +
      "daily format structure (i.e. trainDir/2017/01/20/[input data files]).")

  val validationDataDaysRange: Param[DaysRange] = ParamUtils.createParam[DaysRange](
    "validation data days range",
    "Inclusive date range for validation data, computed from a range of days prior to today. If specified, the " +
      "validation directories are expected to be in the daily format structure (i.e. " +
      "trainDir/2017/01/20/[input data files]).")

  val minValidationPartitions: Param[Int] = ParamUtils.createParam[Int](
    "minimum validation partitions",
    "Minimum number of partitions for the validation data (if any).",
    ParamValidators.gt[Int](0.0))

  val partialRetrainModelDirectory: Param[Path] = ParamUtils.createParam(
    "partial retrain model directory",
    "Path to directory containing a model to use as a base for partial retraining.")

  val partialRetrainLockedCoordinates: Param[Set[CoordinateId]] = ParamUtils.createParam(
    "partial retrain locked coordinates",
    "The set of coordinates present in the pre-trained model to reuse during partial retraining.",
    PhotonParamValidators.nonEmpty)

  val outputMode: Param[ModelOutputMode] = ParamUtils.createParam[ModelOutputMode](
    "output mode",
    "Granularity of model output to HDFS.")

  val coordinateConfigurations: Param[Map[CoordinateId, CoordinateConfiguration]] =
    ParamUtils.createParam[Map[CoordinateId, CoordinateConfiguration]](
      "coordinate configurations",
      "A map of coordinate names to configurations.",
      PhotonParamValidators.nonEmpty[TraversableOnce, (CoordinateId, CoordinateConfiguration)])

  val coordinateUpdateSequence: Param[Seq[CoordinateId]] = ParamUtils.createParam(
    "coordinate update sequence",
    "The order in which coordinates are updated by the descent algorithm. It is recommended to order coordinates by " +
      "their stability (i.e. by looking at the variance of the feature distribution [or correlation with labels] for " +
      "each coordinate).",
    PhotonParamValidators.nonEmpty)

  val coordinateDescentIterations: Param[Int] = ParamUtils.createParam[Int](
    "coordinate descent iterations",
    "Number of coordinate descent iterations (one iteration is one full traversal of the coordinate update sequence).",
    ParamValidators.gt[Int](0.0))

  val normalization: Param[NormalizationType] = ParamUtils.createParam[NormalizationType](
    "normalization",
    "Type of normalization to use during training.")

  val dataSummaryDirectory: Param[Path] = ParamUtils.createParam[Path](
    "data summary directory",
    "Path to record optional summary statistics about the training data.")

  val treeAggregateDepth: Param[Int] = ParamUtils.createParam[Int](
    "tree aggregate depth",
    "Suggested depth for tree aggregation.",
    ParamValidators.gt[Int](0.0))

  val hyperParameterTuning: Param[HyperparameterTuningMode] = ParamUtils.createParam[HyperparameterTuningMode](
    "hyper parameter tuning",
    "Type of automatic hyper-parameter tuning to perform during training.")

  val hyperParameterTuningIter: Param[Int] = ParamUtils.createParam[Int](
    "hyper parameter tuning iterations",
    "Number of iterations of hyper-parameter tuning to perform (if enabled).",
    ParamValidators.gt[Int](0.0))

  val hyperParameterTuningRange: Param[DoubleRange] = ParamUtils.createParam[DoubleRange](
    "hyper parameter tuning range",
    "Range of values within which to search for the optimal hyper-parameters (if enabled).",
    (range: DoubleRange) => range.start > 0.0)

  val computeVariance: Param[Boolean] = ParamUtils.createParam[Boolean](
    "compute variance",
    "Whether to compute the coefficient variances.")

  val useWarmStart: Param[Boolean] = ParamUtils.createParam[Boolean](
    "use warm start",
    "Whether to re-use trained GAME models as starting points.")

  val modelSparsityThreshold: Param[Double] = ParamUtils.createParam[Double](
    "model sparsity threshold",
    "The model sparsity threshold, or the minimum absolute value considered nonzero when persisting a model",
    ParamValidators.gt[Double](0.0))

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
  // Params functions
  //

  /**
   * Check that all required parameters have been set and validate interactions between parameters.
   */
  override def validateParams(paramMap: ParamMap = extractParamMap): Unit = {

    super.validateParams(paramMap)

    // Just need to check that these parameters are explicitly set
    paramMap(trainingTask)
    paramMap(coordinateDescentIterations)

    val coordinateConfigs = paramMap(coordinateConfigurations)
    val updateSequence = paramMap(coordinateUpdateSequence)
    val featureShards = paramMap(featureShardConfigurations)
    val retrainModelDirOpt = paramMap.get(partialRetrainModelDirectory)
    val retrainModelCoordsOpt = paramMap.get(partialRetrainLockedCoordinates)
    val normalizationType = paramMap.getOrElse(normalization, getOrDefault(normalization))
    val hyperparameterTuningMode = paramMap.getOrElse(hyperParameterTuning, getOrDefault(hyperParameterTuning))

    // If partial retraining is enabled, both a model to use and list of coordinates to reuse must be provided
    val coordinatesToTrain = (retrainModelDirOpt, retrainModelCoordsOpt) match {
      case (Some(_), Some(retrainModelCoords)) =>

        // Locked coordinates should not be present in the coordinate configurations
        require(
          coordinateConfigs.keys.forall(coordinateId => !retrainModelCoords.contains(coordinateId)),
          "One or more locked coordinates for partial retraining are present in the coordinate configurations.")

        val newCoordinates = updateSequence.filterNot(retrainModelCoords.contains)

        // No point in training if every coordinate is being reused
        require(
          newCoordinates.nonEmpty,
          "All coordinates in the update sequence are locked coordinates from the pre-trained model: no new " +
            "coordinates to train.")

        // All locked coordinates must be used by the update sequence
        require(
          retrainModelCoords.forall(updateSequence.contains),
          "One or more locked coordinates for partial retraining are missing from the update sequence.")

        newCoordinates

      case (None, None) =>
        updateSequence

      case (Some(_), None) =>
        throw new IllegalArgumentException("Missing locked coordinates for partial retraining.")

      case (None, Some(_)) =>
        throw new IllegalArgumentException("Partial retraining coordinates provided without model.")
    }

    // Each (non-reused) coordinate in the update sequence must have a configuration
    coordinatesToTrain.foreach { coordinate =>
      require(
        coordinateConfigs.contains(coordinate),
        s"Coordinate '$coordinate' in the update sequence is missing configuration.")
    }

    // Each feature shard used by a coordinate must have a configuration
    coordinateConfigs.foreach { case(coordinate, config) =>

      val featureShardId = config.dataConfiguration.featureShardId

      require(
        featureShards.contains(featureShardId),
        s"Feature shard '$featureShardId' used by coordinate '$coordinate' is missing from the set of column names")
    }

    // If standardization is enabled, all feature shards must have intercepts enabled
    normalizationType match {
      case NormalizationType.STANDARDIZATION =>
        require(
          featureShards.forall(_._2.hasIntercept),
          "Standardization enabled; all feature shards must have intercept enabled.")

      case _ =>
    }

    // If hyperparameter tuning is enabled, need to specify the number of tuning iterations
    hyperparameterTuningMode match {
      case HyperparameterTuningMode.BAYESIAN | HyperparameterTuningMode.RANDOM =>
        require(
          paramMap.get(hyperParameterTuningIter).isDefined,
          "Hyperparameter tuning enabled, but number of iterations unspecified.")
      case _ =>
    }
  }

  /**
   * Set default values for parameters that have them.
   */
  private def setDefaultParams(): Unit = {

    setDefault(inputColumnNames, InputColumnsNames())
    setDefault(minValidationPartitions, 1)
    setDefault(outputMode, ModelOutputMode.BEST)
    setDefault(overrideOutputDirectory, false)
    setDefault(normalization, NormalizationType.NONE)
    setDefault(hyperParameterTuning, HyperparameterTuningMode.NONE)
    setDefault(hyperParameterTuningRange, DoubleRange(1e-4, 1e4))
    setDefault(computeVariance, false)
    setDefault(useWarmStart, false)
    setDefault(dataValidation, DataValidationType.VALIDATE_DISABLED)
    setDefault(logLevel, PhotonLogger.LogLevelInfo)
    setDefault(applicationName, DEFAULT_APPLICATION_NAME)
    setDefault(modelSparsityThreshold, VectorUtils.DEFAULT_SPARSITY_THRESHOLD)
  }

  /**
   * Clear all set parameters.
   */
  def clear(): Unit = params.foreach(clear)

  //
  // Training driver functions
  //

  /**
   * Prepare the training data, fit models and select best model.
   * There is one model for each combination of fixed and random effect specified in the params.
   *
   * @note all intercept terms are turned ON by default in prepareFeatureMaps.
   */
  protected[training] def run(): Unit = {

    validateParams()

    Timed("Clean output directories") {
      cleanOutputDirs()
    }

    val featureIndexMapLoadersOpt = Timed("Prepare features") {
      prepareFeatureMaps()
    }
    val (trainingData, featureIndexMapLoaders) = Timed(s"Read training data") {
      readTrainingData(featureIndexMapLoadersOpt)
    }
    val validationData = Timed(s"Read validation data") {
      readValidationData(featureIndexMapLoaders)
    }
    val (partialRetrainingModelOpt, partialRetrainingDataConfigsOpt) = Timed("Load model for partial retraining") {
      (get(partialRetrainModelDirectory), get(partialRetrainLockedCoordinates)) match {
        case (Some(preTrainedModelDir), Some(lockedCoordinates)) =>
          val (gameModel, modelIndexMapLoaders) = ModelProcessingUtils
            .loadGameModelFromHDFS(
              sc,
              preTrainedModelDir,
              StorageLevel.VERY_FREQUENT_REUSE_RDD_STORAGE_LEVEL,
              Some(featureIndexMapLoaders))

          // All of the locked coordinates must be present in the pre-trained model
          require(
            lockedCoordinates.forall(gameModel.toMap.contains),
            "One or more locked coordinates for partial retraining are missing from the pre-trained model.")

          // The feature shards for the locked coordinates must be defined
          modelIndexMapLoaders
            .keys
            .filter(lockedCoordinates.contains)
            .foreach { featureShard =>
              require(
                featureIndexMapLoaders.contains(featureShard),
                s"Missing feature shard definition for shard '$featureShard' used by the pre-trained model.")
            }

          val dataConfigs = gameModel
            .toMap
            .filter { case (coordinateId, _) =>
              lockedCoordinates.contains(coordinateId)
            }
            .mapValues {
              case fEM: FixedEffectModel =>
                FixedEffectDataConfiguration(fEM.featureShardId)

              case rEM: RandomEffectModel =>
                RandomEffectDataConfiguration(
                  rEM.randomEffectType,
                  rEM.featureShardId,
                  projectorType = IdentityProjection)

              case other: DatumScoringModel =>
                throw new IllegalArgumentException(s"Encountered unknown model type '${other.getClass.getName}'")
            }
            .map(identity)

          (Some(gameModel), Some(dataConfigs))

        case _ =>
          (None, None)
      }
    }

    Timed("Validate data") {
      DataValidators.sanityCheckDataFrameForTraining(
        trainingData,
        getRequiredParam(trainingTask),
        getOrDefault(dataValidation),
        getOrDefault(inputColumnNames),
        getRequiredParam(featureShardConfigurations).keySet)

      validationData match {
        case Some(x) => DataValidators.sanityCheckDataFrameForTraining(
          x,
          getRequiredParam(trainingTask),
          getOrDefault(dataValidation),
          getOrDefault(inputColumnNames),
          getRequiredParam(featureShardConfigurations).keySet)
        case None => None
      }
    }

    val featureShardStats = Timed("Calculate statistics for each feature shard") {
      calculateAndSaveFeatureShardStats(trainingData, featureIndexMapLoaders)
    }

    val normalizationContexts = Timed("Prepare normalization contexts") {
      prepareNormalizationContexts(trainingData, featureIndexMapLoaders, featureShardStats)
        .map(_.mapValues(context => PhotonBroadcast(sc.broadcast(context))))
    }

    val gameOptimizationConfigs = Timed("Prepare optimization configuration(s)") {
      prepareGameOptConfigs(getRequiredParam(coordinateConfigurations))
    }

    val gameEstimator = Timed("Setup estimator") {

      val coordinateDataConfigs = getRequiredParam(coordinateConfigurations).mapValues(_.dataConfiguration) ++
        partialRetrainingDataConfigsOpt.getOrElse(Map())

      // Set estimator parameters
      val estimator = new GameEstimator(sc, logger)
        .setTrainingTask(getRequiredParam(trainingTask))
        .setCoordinateDataConfigurations(coordinateDataConfigs)
        .setCoordinateUpdateSequence(getRequiredParam(coordinateUpdateSequence))
        .setCoordinateDescentIterations(getRequiredParam(coordinateDescentIterations))
        .setComputeVariance(getOrDefault(computeVariance))
        .setWarmStart(getOrDefault(useWarmStart))

      get(inputColumnNames).foreach(estimator.setInputColumnNames)
      partialRetrainingModelOpt.foreach(estimator.setPartialRetrainModel)
      get(partialRetrainLockedCoordinates).foreach(estimator.setPartialRetrainLockedCoordinates)
      normalizationContexts.foreach(estimator.setCoordinateNormalizationContexts)
      get(treeAggregateDepth).foreach(estimator.setTreeAggregateDepth)
      get(evaluators).foreach(estimator.setValidationEvaluators)

      estimator
    }

    val explicitModels = Timed("Fit models") {
      gameEstimator.fit(trainingData, validationData, gameOptimizationConfigs)
    }

    val tunedModels = Timed("Tune hyper-parameters") {
      runHyperparameterTuning(gameEstimator, trainingData, validationData, explicitModels)
    }

    val (outputModels, bestModel) = selectModels(explicitModels, tunedModels)

    Timed("Save models") {
      saveModelToHDFS(featureIndexMapLoaders, outputModels, bestModel)
    }
  }

  /**
   * Clean up the directories in which we are going to output the models.
   */
  private def cleanOutputDirs(): Unit = {

    val configuration = sc.hadoopConfiguration
    val overrideOutputDir = getOrDefault(overrideOutputDirectory)

    IOUtils.processOutputDir(getRequiredParam(rootOutputDirectory), overrideOutputDir, configuration)
    get(dataSummaryDirectory).foreach(IOUtils.processOutputDir(_, overrideOutputDir, configuration))
  }

  /**
   * Reads the training dataset, handling specifics of input date ranges in the params.
   *
   * @param featureIndexMapLoadersOpt Optional feature index map loaders
   * @return A ([[DataFrame]] of input data, feature index map loaders) pair
   */
  private def readTrainingData(featureIndexMapLoadersOpt: Option[Map[FeatureShardId, IndexMapLoader]])
    : (DataFrame, Map[FeatureShardId, IndexMapLoader]) = {

    val dateRangeOpt = IOUtils.resolveRange(get(inputDataDateRange), get(inputDataDaysRange))
    val trainingRecordsPath = pathsForDateRange(getRequiredParam(inputDataDirectories), dateRangeOpt)

    logger.debug(s"Training records paths:\n${trainingRecordsPath.mkString("\n")}")

    val numPartitions = getRequiredParam(coordinateConfigurations).values.map(_.dataConfiguration.minNumPartitions).max

    // The 'map(identity)' call is required due to a long-standing Scala bug SI-7005: the result of a 'mapValues' call
    // is not serializable.
    new AvroDataReader(sc)
      .readMerged(
        trainingRecordsPath.map(_.toString),
        featureIndexMapLoadersOpt,
        getRequiredParam(featureShardConfigurations),
        numPartitions)
  }

  /**
   * Reads the validation data set, handling specifics of input date ranges in the params.
   *
   * @param featureIndexMapLoaders The feature index map loaders
   * @return The loaded data frame
   */
  private def readValidationData(featureIndexMapLoaders: Map[FeatureShardId, IndexMapLoader]): Option[DataFrame] =
    get(validationDataDirectories).map { validationDirs =>

      val dateRange = IOUtils.resolveRange(get(validationDataDateRange), get(validationDataDaysRange))
      val validationRecordsPath = pathsForDateRange(validationDirs, dateRange)

      logger.debug(s"Validation records paths:\n${validationRecordsPath.mkString("\n")}")

      // The 'map(identity)' call is required due to a long-standing Scala bug SI-7005: the result of a 'mapValues' call
      // is not serializable.
      new AvroDataReader(sc)
        .readMerged(
          validationRecordsPath.map(_.toString),
          featureIndexMapLoaders,
          getRequiredParam(featureShardConfigurations),
          getOrDefault(minValidationPartitions))
    }

  /**
   * Prepare normalization contexts, if normalization is enabled.
   *
   * @param trainingData The training data
   * @param featureIndexMapLoaders The index map loaders
   * @return Normalization contexts for each coordinate, or None if normalization is disabled
   */
  private def prepareNormalizationContexts(
      trainingData: DataFrame,
      featureIndexMapLoaders: IndexMapLoaders,
      statistics: FeatureShardStatisticsOpt): Option[Map[CoordinateId, NormalizationContext]] =

    Utils.filter(getOrDefault(normalization) != NormalizationType.NONE) {
      val featureShardToNormalizationContextMap = statistics
        .getOrElse(calculateStatistics(trainingData, featureIndexMapLoaders.keys))
        .map { case (featureShardId, featureShardStats) =>
          val intercept = featureIndexMapLoaders(featureShardId).indexMapForDriver().get(Constants.INTERCEPT_KEY)
          (featureShardId, NormalizationContext(getOrDefault(normalization), featureShardStats, intercept))
        }
        .toMap

      getRequiredParam(coordinateConfigurations).mapValues { coordinateConfig =>
        featureShardToNormalizationContextMap(coordinateConfig.dataConfiguration.featureShardId)
      }
    }

  /**
   * Compute basic statistics (same as spark-ml) of the training data for each feature shard.
   * At the same time, save those statistics to disk.
   *
   * @param trainingData The training data
   * @param featureIndexMapLoaders The index map loaders
   * @return Basic for each feature shard
   */
  private def calculateAndSaveFeatureShardStats(
      trainingData: DataFrame,
      featureIndexMapLoaders: IndexMapLoaders): FeatureShardStatisticsOpt =
    get(dataSummaryDirectory).map { summarizationOutputDir: Path =>
      calculateStatistics(trainingData, featureIndexMapLoaders.keys)
        .tap { case (featureShardId, featureShardStats) =>
          val outputPath = new Path(summarizationOutputDir, featureShardId)
          val indexMap = featureIndexMapLoaders(featureShardId).indexMapForDriver()

          ModelProcessingUtils.writeBasicStatistics(sc, featureShardStats, outputPath, indexMap)
        }
    }

  /**
   * Calculate basic statistics (same as spark-ml) on a DataFrame.
   *
   * @param data The data to compute statistics on
   * @param featureShards The feature shards for which to compute statistics
   * @return One BasicStatisticalSummary per feature shard
   */
  private def calculateStatistics(
      data: DataFrame,
      featureShards: Iterable[FeatureShardId]): FeatureShardStatistics =
    featureShards.map { featureShardId =>
      // Calling rdd explicitly here to avoid a typed encoder lookup in Spark 2.1
      (featureShardId, BasicStatisticalSummary(
        data.select(featureShardId)
          .rdd
          .map(_.getAs[SparkMLVector](0))))
    }

  /**
   * Expand each [[CoordinateOptimizationConfiguration]] for its regularization values, then compute all permutations of
   * the coordinates.
   *
   * Ex. Map(a -> (1, 2), b -> (1, 2)) => (Map(a -> 1, b -> 1), Map(a -> 1, b -> 2), Map(a -> 2, b -> 1), Map(a -> 2, b -> 2))
   *
   * @param coordinateConfigurations The coordinate definitions
   * @return A list of [[GameOptimizationConfiguration]], for each of a which a model will be trained.
   */
  private def prepareGameOptConfigs(coordinateConfigurations: Map[CoordinateId, CoordinateConfiguration])
    : Seq[GameOptimizationConfiguration] =

    coordinateConfigurations
      .mapValues(_.expandOptimizationConfigurations)
      .foldLeft(Seq[GameOptimizationConfiguration](Map())) { case (partialGameOptConfigs, (coordinateId, coordinateOptConfigs)) =>
        for (map <- partialGameOptConfigs; newConfig <- coordinateOptConfigs) yield {
          map + ((coordinateId, newConfig))
        }
      }

  /**
   * Run hyperparameter tuning to produce models with automatically-tuned hyperparameters
   *
   * @param estimator The estimator to use for training and validation
   * @param trainingData The training data
   * @param validationData The validation data
   * @param models The previously trained and evaluated models
   */
  private def runHyperparameterTuning(
      estimator: GameEstimator,
      trainingData: DataFrame,
      validationData: Option[DataFrame],
      models: Seq[GameEstimator.GameResult]): Seq[GameEstimator.GameResult] =

    validationData match {
      case Some(testData) if getOrDefault(hyperParameterTuning) != HyperparameterTuningMode.NONE =>

        // TODO: Match on this to make it clearer
        val evaluator = models.head._2.get.head._1
        val baseConfig = models.head._3
        val evaluationFunction = new GameEstimatorEvaluationFunction(estimator, baseConfig, trainingData, testData)

        val range = getOrDefault(hyperParameterTuningRange)
        val ranges = List.fill(evaluationFunction.numParams)(DoubleRange(log(range.start), log(range.end)))

        val searcher = getOrDefault(hyperParameterTuning) match {
          case HyperparameterTuningMode.BAYESIAN =>
            new GaussianProcessSearch[GameEstimator.GameResult](ranges, evaluationFunction, evaluator)

          case HyperparameterTuningMode.RANDOM =>
            new RandomSearch[GameEstimator.GameResult](ranges, evaluationFunction)
        }

        searcher.find(getOrDefault(hyperParameterTuningIter), models)

      case _ => Seq()
    }

  /**
   * Select which models will be output to HDFS.
   *
   * @param explicitModels Models trained using combinations of explicitly specified regularization weights
   * @param tunedModels Models trained during automatic hyperparameter tuning
   * @return The set of models to output to HDFS
   */
  protected[training] def selectModels(
      explicitModels: Seq[GameEstimator.GameResult],
      tunedModels: Seq[GameEstimator.GameResult]): (Seq[GameEstimator.GameResult], Option[GameEstimator.GameResult]) = {

    val allModels = explicitModels ++ tunedModels
    val modelOutputMode = getOrDefault(outputMode)
    val outputModels = modelOutputMode match {
      case ModelOutputMode.NONE | ModelOutputMode.BEST => Seq()
      case ModelOutputMode.EXPLICIT => explicitModels
      case ModelOutputMode.TUNED => tunedModels
      case ModelOutputMode.ALL => allModels
      case other => throw new IllegalArgumentException(s"Unknown output mode: $other")
    }
    val bestModel = modelOutputMode match {
      case ModelOutputMode.NONE => None
      case _ => Timed("Select best model") { Some(selectBestModel(allModels)) }
    }

    (outputModels, bestModel)
  }

  /**
   * Select best model according to validation evaluator.
   *
   * @param models The models to evaluate (single evaluator, on the validation data set)
   * @return The best model
   */
  protected[training] def selectBestModel(models: Seq[GameEstimator.GameResult]): GameEstimator.GameResult = {

    val bestResult = models
      .flatMap { case (model, evaluations, modelConfig) => evaluations.map((model, _, modelConfig)) }
      .reduceOption { (configModelEval1, configModelEval2) =>
        val (eval1, eval2) = (configModelEval1._2, configModelEval2._2)
        val (evaluator1, score1) = eval1.head
        val (evaluator2, score2) = eval2.head

        // TODO: Each iteration of the Bayesian hyper-parameter tuning recomputes the GAME data set. This causes the
        // TODO: equality check to fail: not only are the evaluators not identical (ev1.eq(ev2)) but they're not equal
        // TODO: either (they reference different RDDs, which are computed identically). The below check is a temporary
        // TODO: solution.
        require(
          evaluator1.evaluatorType == evaluator2.evaluatorType,
          "Evaluator mismatch while selecting best model; some error has occurred during validation.")

        if (evaluator1.betterThan(score1, score2)) configModelEval1 else configModelEval2
      }

    bestResult match {
      case Some(gameResult) =>
        val (model, eval, configs) = gameResult

        logger.info(s"Best model has ${eval.head._1.getEvaluatorName} score of ${eval.head._2} and following config:")
        logger.info(IOUtils.optimizationConfigToString(configs))

        // Computing model summary is slow, we should only do it if necessary
        if (logger.isDebugEnabled) {
          logger.debug(s"Model summary:\n${model.toSummaryString}\n")
        }

        (model, Some(eval), configs)

      case None =>
        logger.info("Could not select best model: missing evaluation results. Using most recently trained model.")
        models.last
    }
  }

  /**
   * Write the GAME models to HDFS.
   *
   * TODO: Deprecate model-spec then remove it in favor of model-metadata, but there are clients!
   *
   * @param featureShardIdToFeatureMapLoader The shard ids
   * @param models All the models that were producing during training
   * @param bestModel The best model
   */
  private def saveModelToHDFS(
      featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader],
      models: Seq[GameEstimator.GameResult],
      bestModel: Option[GameEstimator.GameResult]): Unit =

    if (getOrDefault(outputMode) != ModelOutputMode.NONE) {

      val hadoopConfiguration = sc.hadoopConfiguration
      val rootOutputDir = getRequiredParam(rootOutputDirectory)
      val allOutputDir = new Path(rootOutputDir, MODELS_DIR)
      val task = getRequiredParam(trainingTask)
      val REMFileLimit = get(outputFilesLimit)

      // Write the best model to HDFS
      bestModel match {
        case Some((model, _, modelConfig)) =>

          val modelOutputDir = new Path(rootOutputDir, BEST_MODEL_DIR)
          val modelSpecDir = new Path(modelOutputDir, MODEL_SPEC_DIR)

          Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)
          IOUtils.writeOptimizationConfigToHDFS(
            modelConfig,
            modelSpecDir,
            hadoopConfiguration,
            forceOverwrite = false)
          ModelProcessingUtils.saveGameModelToHDFS(
            sc,
            modelOutputDir,
            model,
            task,
            modelConfig,
            REMFileLimit,
            featureShardIdToFeatureMapLoader,
            getOrDefault(modelSparsityThreshold))

          logger.info("Saved best model to HDFS")

        case None =>
          logger.info("No best model to save to HDFS")
      }

      // Write additional models to HDFS
      models.foldLeft(0) {
        case (modelIndex, (model, _, modelConfig)) =>

          val modelOutputDir = new Path(allOutputDir, modelIndex.toString)
          val modelSpecDir = new Path(modelOutputDir, MODEL_SPEC_DIR)

          Utils.createHDFSDir(modelOutputDir, hadoopConfiguration)
          IOUtils.writeOptimizationConfigToHDFS(
            modelConfig,
            modelSpecDir,
            hadoopConfiguration,
            forceOverwrite = false)
          ModelProcessingUtils.saveGameModelToHDFS(
            sc,
            modelOutputDir,
            model,
            task,
            modelConfig,
            REMFileLimit,
            featureShardIdToFeatureMapLoader,
            getOrDefault(modelSparsityThreshold))

          modelIndex + 1
      }
    }

  /**
   * Entry point to the driver.
   *
   * @param args The command line arguments for the job
   */
  def main(args: Array[String]): Unit = {

    // Parse and apply parameters
    val params: ParamMap = ScoptGameTrainingParametersParser.parseFromCommandLine(args)
    params.toSeq.foreach(set)

    sc = SparkContextConfiguration.asYarnClient(getOrDefault(applicationName), useKryo = true)
    logger = new PhotonLogger(new Path(getRequiredParam(rootOutputDirectory), LOGS), sc)
    logger.setLogLevel(getOrDefault(logLevel))

    try {
      Timed("Total time in training Driver")(run())

    } catch { case e: Exception =>
      logger.error("Failure while running the driver", e)
      throw e

    } finally {
      logger.close()
      sc.stop()
    }
  }
}
