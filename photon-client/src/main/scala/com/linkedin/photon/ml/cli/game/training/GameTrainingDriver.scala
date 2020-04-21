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

import org.apache.commons.cli.MissingArgumentException
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{Vector => SparkMLVector}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators, Params}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml._
import com.linkedin.photon.ml.HyperparameterTunerName.HyperparameterTunerName
import com.linkedin.photon.ml.HyperparameterTuningMode.HyperparameterTuningMode
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types._
import com.linkedin.photon.ml.cli.game.GameDriver
import com.linkedin.photon.ml.data.{DataValidators, FixedEffectDataConfiguration, InputColumnsNames, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.data.avro.{AvroDataReader, ModelProcessingUtils}
import com.linkedin.photon.ml.estimators.GameEstimator.GameOptimizationConfiguration
import com.linkedin.photon.ml.estimators.{GameEstimator, GameEstimatorEvaluationFunction}
import com.linkedin.photon.ml.hyperparameter.tuner.HyperparameterTunerFactory
import com.linkedin.photon.ml.index.{IndexMap, IndexMapLoader}
import com.linkedin.photon.ml.io.{CoordinateConfiguration, ModelOutputMode, RandomEffectCoordinateConfiguration}
import com.linkedin.photon.ml.io.ModelOutputMode.ModelOutputMode
import com.linkedin.photon.ml.io.scopt.game.ScoptGameTrainingParametersParser
import com.linkedin.photon.ml.model.{DatumScoringModel, FixedEffectModel, RandomEffectModel}
import com.linkedin.photon.ml.normalization.NormalizationType.NormalizationType
import com.linkedin.photon.ml.normalization.{NormalizationContext, NormalizationType}
import com.linkedin.photon.ml.optimization.{RegularizationType, VarianceComputationType}
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.optimization.game.{CoordinateOptimizationConfiguration, GLMOptimizationConfiguration}
import com.linkedin.photon.ml.stat.FeatureDataStatistics
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

  type FeatureShardStatistics = Iterable[(FeatureShardId, FeatureDataStatistics)]
  type FeatureShardStatisticsOpt = Option[FeatureShardStatistics]
  type IndexMapLoaders = Map[FeatureShardId, IndexMapLoader]

  //
  // Members
  //

  private val DEFAULT_APPLICATION_NAME = "GAME-Training"

  protected[training] val MODELS_DIR = "models"
  protected[training] val MODEL_SPEC_DIR = "model-spec"
  protected[training] val BEST_MODEL_DIR = "best"
  protected[training] val GROUP_EVAL_DIR = "group-eval"

  protected[training] var sparkSession: SparkSession = _
  protected[training] var sc: SparkContext = _
  protected[training] implicit var logger: PhotonLogger = _

  //
  // Parameters
  //

  val trainingTask: Param[TaskType] = ParamUtils.createParam(
    "training task",
    "The type of training task to perform.")

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

  val outputMode: Param[ModelOutputMode] = ParamUtils.createParam[ModelOutputMode](
    "output mode",
    "Granularity of model output to HDFS.")

  val partialRetrainLockedCoordinates: Param[Set[CoordinateId]] = ParamUtils.createParam(
    "partial retrain locked coordinates",
    "The set of coordinates present in the pre-trained model to reuse during partial retraining.",
    PhotonParamValidators.nonEmpty)

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

  val hyperParameterTunerName: Param[HyperparameterTunerName] = ParamUtils.createParam[HyperparameterTunerName](
    "hyper parameter tuner",
    "Package name of hyperparameter tuner."
  )

  val hyperParameterTuning: Param[HyperparameterTuningMode] = ParamUtils.createParam[HyperparameterTuningMode](
    "hyper parameter tuning",
    "Type of automatic hyperparameter tuning to perform during training.")

  val hyperParameterTuningIter: Param[Int] = ParamUtils.createParam[Int](
    "hyper parameter tuning iterations",
    "Number of iterations of hyperparameter tuning to perform (if enabled).",
    ParamValidators.gt[Int](0.0))

  val varianceComputationType: Param[VarianceComputationType] = ParamUtils.createParam[VarianceComputationType](
    "variance computation type",
    "Whether to compute coefficient variances and, if so, how.")

  val modelSparsityThreshold: Param[Double] = ParamUtils.createParam[Double](
    "model sparsity threshold",
    "The model sparsity threshold, or the minimum absolute value considered nonzero when persisting a model",
    ParamValidators.gt[Double](0.0))

  val ignoreThresholdForNewModels: Param[Boolean] = ParamUtils.createParam[Boolean](
    "ignore threshold for new models",
    "Flag to ignore the random effect samples lower bound when encountering a random effect ID without an " +
      "existing model during warm-start training.")

  val incrementalTraining: Param[Boolean] = ParamUtils.createParam[Boolean](
    "incremental training",
    "Flag to enable incremental training.")

  val savePerGroupEvaluationResult: Param[Boolean] = ParamUtils.createParam[Boolean](
    "save per-group evaluation result",
    "Flag to save group evaluation result for random effect in a separate output file.")

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

    setDefault(inputColumnNames, InputColumnsNames())
    setDefault(minValidationPartitions, 1)
    setDefault(outputMode, ModelOutputMode.BEST)
    setDefault(overrideOutputDirectory, false)
    setDefault(normalization, NormalizationType.NONE)
    setDefault(hyperParameterTunerName, HyperparameterTunerName.DUMMY)
    setDefault(hyperParameterTuning, HyperparameterTuningMode.NONE)
    setDefault(varianceComputationType, VarianceComputationType.NONE)
    setDefault(dataValidation, DataValidationType.VALIDATE_DISABLED)
    setDefault(logLevel, PhotonLogger.LogLevelInfo)
    setDefault(applicationName, DEFAULT_APPLICATION_NAME)
    setDefault(modelSparsityThreshold, VectorUtils.DEFAULT_SPARSITY_THRESHOLD)
    setDefault(timeZone, Constants.DEFAULT_TIME_ZONE)
    setDefault(ignoreThresholdForNewModels, false)
    setDefault(incrementalTraining, false)
    setDefault(savePerGroupEvaluationResult, false)
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

    super.validateParams(paramMap)

    // Just need to check that these parameters are explicitly set
    paramMap(trainingTask)
    paramMap(coordinateDescentIterations)

    val coordinateConfigs = paramMap(coordinateConfigurations)
    val updateSequence = paramMap(coordinateUpdateSequence)
    val featureShards = paramMap(featureShardConfigurations)
    val baseModelDirOpt = paramMap.get(modelInputDirectory)
    val retrainModelCoordsOpt = paramMap.get(partialRetrainLockedCoordinates)
    val inputColNames = paramMap.getOrElse(inputColumnNames, getDefault(inputColumnNames).get).getNames
    val normalizationType = paramMap.getOrElse(normalization, getDefault(normalization).get)
    val hyperParameterTuningMode = paramMap.getOrElse(hyperParameterTuning, getDefault(hyperParameterTuning).get)
    val ignoreThreshold = paramMap.getOrElse(ignoreThresholdForNewModels, getDefault(ignoreThresholdForNewModels).get)
    val isIncrementalTraining = paramMap.getOrElse(incrementalTraining, getDefault(incrementalTraining).get)

    // Partial retraining and warm-start training require an initial GAME model to be provided as input
    val coordinatesToTrain = (baseModelDirOpt, retrainModelCoordsOpt) match {
      case (Some(_), Some(retrainModelCoords)) =>

        // Locked coordinates should not be present in the coordinate configurations
        require(
          coordinateConfigs.keys.forall(coordinateId => !retrainModelCoords.contains(coordinateId)),
          "One or more locked coordinates for partial retraining are present in the coordinate configurations.")

        // All locked coordinates must be used by the update sequence
        require(
          retrainModelCoords.forall(updateSequence.contains),
          "One or more locked coordinates for partial retraining are missing from the update sequence.")

        val newCoordinates = updateSequence.filterNot(retrainModelCoords.contains)

        // No point in training if every coordinate is being reused
        require(
          newCoordinates.nonEmpty,
          "All coordinates in the update sequence are locked coordinates from the pre-trained model: no new " +
            "coordinates to train.")

        newCoordinates

      case (Some(_), None) | (None, None) =>
        updateSequence

      case (None, Some(_)) =>
        throw new IllegalArgumentException("Partial retraining enabled, but no base model provided.")
    }

    // Each (non-reused) coordinate in the update sequence must have a configuration
    coordinatesToTrain.foreach { coordinate =>
      require(
        coordinateConfigs.contains(coordinate),
        s"Coordinate '$coordinate' in the update sequence is missing configuration.")
    }

    // Each feature shard used by a coordinate must have a configuration
    coordinateConfigs.foreach { case (coordinate, config) =>

      val featureShardId = config.dataConfiguration.featureShardId

      require(
        featureShards.contains(featureShardId),
        s"Feature shard '$featureShardId' used by coordinate '$coordinate' is missing from the set of column names")

      config match {
        case reConfig: RandomEffectCoordinateConfiguration =>
          val randomEffectType = reConfig.dataConfiguration.randomEffectType

          require(
            !inputColNames.contains(randomEffectType),
            s"Cannot use field '$randomEffectType' as random effect grouping field for coordinate '$coordinate'; " +
              "that field is reserved")

        case _ =>
      }
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
    hyperParameterTuningMode match {
      case HyperparameterTuningMode.BAYESIAN | HyperparameterTuningMode.RANDOM =>
        require(
          paramMap.get(hyperParameterTuningIter).isDefined,
          "Hyperparameter tuning enabled, but number of iterations unspecified.")

      case _ =>
    }

    // Warm-start must be enabled to ignore threshold
    require(
      !ignoreThreshold || baseModelDirOpt.isDefined,
      s"'${ignoreThresholdForNewModels.name}' set but no initial model provided (warm-start not enabled).")

    // If incremental training is enabled, prior model must be defined.
    require(
      !isIncrementalTraining || baseModelDirOpt.isDefined,
      s"'${incrementalTraining.name}' set but no initial model provided.")
  }

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

    val updateSequence = getRequiredParam(coordinateUpdateSequence)

    val avroDataReader = new AvroDataReader()
    val featureIndexMapLoadersOpt = Timed("Prepare features") {
      prepareFeatureMaps()
    }
    val (trainingData, featureIndexMapLoaders) = Timed(s"Read training data") {
      readTrainingData(avroDataReader, featureIndexMapLoadersOpt)
    }
    val validationData = Timed(s"Read validation data") {
      readValidationData(avroDataReader, featureIndexMapLoaders)
    }

    val interceptIndices = featureIndexMapLoaders.flatMap { case (coordinateId, indexMap) =>
      indexMap.indexMapForDriver().getIndex(Constants.INTERCEPT_KEY) match {
        case IndexMap.NULL_KEY => None
        case i if i >= 0 => Some(coordinateId, i)
        case _ => throw new IndexOutOfBoundsException("Intercept Index should not be negative.")
      }
    }

    trainingData.persist(StorageLevel.DISK_ONLY)
    validationData.map(_.persist(StorageLevel.DISK_ONLY))

    val modelOpt = get(modelInputDirectory).map { modelDir =>
      Timed("Load model for warm-start training / incremental learning") {
        ModelProcessingUtils.loadGameModelFromHDFS(
          sc,
          modelDir,
          StorageLevel.MEMORY_AND_DISK,
          featureIndexMapLoaders,
          Some(updateSequence.toSet))
      }
    }

    val partialRetrainingDataConfigsOpt = get(partialRetrainLockedCoordinates).map { lockedCoordinates =>
      Timed("Build data configurations for locked coordinates") {

        val modelMap = modelOpt.get.toMap

        require(
          lockedCoordinates.forall(modelMap.contains),
          "One or more locked coordinates for partial retraining are missing from the initial model.")

        modelMap
          .filter { case (coordinateId, _) =>
            lockedCoordinates.contains(coordinateId)
          }
          .mapValues {
            case fEM: FixedEffectModel =>
              FixedEffectDataConfiguration(fEM.featureShardId)

            case rEM: RandomEffectModel =>
              RandomEffectDataConfiguration(rEM.randomEffectType, rEM.featureShardId)

            case other: DatumScoringModel =>
              throw new IllegalArgumentException(s"Encountered unknown model type '${other.getClass.getName}'")
          }
          .map(identity)
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
    }

    val gameOptimizationConfigs = Timed("Prepare optimization configuration(s)") {
      prepareGameOptConfigs(getRequiredParam(coordinateConfigurations))
    }

    val gameEstimator = Timed("Setup estimator") {

      val coordinateDataConfigs = getRequiredParam(coordinateConfigurations).mapValues(_.dataConfiguration) ++
        partialRetrainingDataConfigsOpt.getOrElse(Map())

      // Set estimator parameters and always use warm start by default
      val estimator = new GameEstimator(sc, logger)
        .setTrainingTask(getRequiredParam(trainingTask))
        .setCoordinateDataConfigurations(coordinateDataConfigs)
        .setCoordinateUpdateSequence(getRequiredParam(coordinateUpdateSequence))
        .setCoordinateDescentIterations(getRequiredParam(coordinateDescentIterations))
        .setCoordinateInterceptIndices(interceptIndices)
        .setVarianceComputation(getOrDefault(varianceComputationType))
        .setIgnoreThresholdForNewModels(getOrDefault(ignoreThresholdForNewModels))
        .setUseWarmStart(true)
        .setIncrementalTraining(getOrDefault(incrementalTraining))
        .setSavePerGroupEvaluationResult(getOrDefault(savePerGroupEvaluationResult))

      get(inputColumnNames).foreach(estimator.setInputColumnNames)
      modelOpt.foreach(estimator.setInitialModel)
      get(partialRetrainLockedCoordinates).foreach(estimator.setPartialRetrainLockedCoordinates)
      normalizationContexts.foreach(estimator.setCoordinateNormalizationContexts)
      get(treeAggregateDepth).foreach(estimator.setTreeAggregateDepth)
      get(evaluators).foreach(estimator.setValidationEvaluators)

      estimator
    }

    val explicitModels = Timed("Fit models") {
      gameEstimator.fit(trainingData, validationData, gameOptimizationConfigs)
    }

    val tunedModels = Timed("Tune hyperparameters") {
      // Disable warm start for autotuning
      gameEstimator.setUseWarmStart(false)
      runHyperparameterTuning(gameEstimator, trainingData, validationData, explicitModels)
    }

    trainingData.unpersist()
    validationData.map(_.unpersist())

    val (outputModels, bestModel) = selectModels(explicitModels, tunedModels)

    Timed("Save models") {
      saveModelToHDFS(featureIndexMapLoaders, outputModels, bestModel)
    }

    if (getOrDefault(savePerGroupEvaluationResult)) {
      Timed("Save per-group evaluation result") {
        savePerGroupEvaluationToHDFS(outputModels)
      }
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
   * @param avroDataReader The [[AvroDataReader]] to use for reading training data
   * @param featureIndexMapLoadersOpt Optional feature index map loaders
   * @return A ([[DataFrame]] of input data, feature index map loaders) pair
   */
  private def readTrainingData(
      avroDataReader: AvroDataReader,
      featureIndexMapLoadersOpt: Option[Map[FeatureShardId, IndexMapLoader]])
    : (DataFrame, Map[FeatureShardId, IndexMapLoader]) = {

    val dateRangeOpt = IOUtils.resolveRange(get(inputDataDateRange), get(inputDataDaysRange), getOrDefault(timeZone))
    val trainingRecordsPath = pathsForDateRange(getRequiredParam(inputDataDirectories), dateRangeOpt)

    logger.debug(s"Training records paths:\n${trainingRecordsPath.mkString("\n")}")

    val numPartitions = getRequiredParam(coordinateConfigurations).values.map(_.dataConfiguration.minNumPartitions).max

    avroDataReader.readMerged(
      trainingRecordsPath.map(_.toString),
      featureIndexMapLoadersOpt,
      getRequiredParam(featureShardConfigurations),
      numPartitions)
  }

  /**
   * Reads the validation dataset, handling specifics of input date ranges in the params.
   *
   * @param avroDataReader The [[AvroDataReader]] to use for reading validation data
   * @param featureIndexMapLoaders The feature index map loaders
   * @return The loaded data frame
   */
  private def readValidationData(
      avroDataReader: AvroDataReader,
      featureIndexMapLoaders: Map[FeatureShardId, IndexMapLoader]): Option[DataFrame] =
    get(validationDataDirectories).map { validationDirs =>

      val dateRange = IOUtils.resolveRange(
        get(validationDataDateRange), get(validationDataDaysRange), getOrDefault(timeZone))
      val validationRecordsPath = pathsForDateRange(validationDirs, dateRange)

      logger.debug(s"Validation records paths:\n${validationRecordsPath.mkString("\n")}")

      // The 'map(identity)' call is required due to a long-standing Scala bug SI-7005: the result of a 'mapValues' call
      // is not serializable.
      avroDataReader.readMerged(
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
        .getOrElse(calculateStatistics(trainingData, featureIndexMapLoaders))
        .map { case (featureShardId, featureShardStats) =>
          (featureShardId, NormalizationContext(getOrDefault(normalization), featureShardStats))
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
      calculateStatistics(trainingData, featureIndexMapLoaders)
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
   * @param featureIndexMapLoaders The index map loaders
   * @return One BasicStatisticalSummary per feature shard
   */
  private def calculateStatistics(
      data: DataFrame,
      featureIndexMapLoaders: IndexMapLoaders): FeatureShardStatistics =
    featureIndexMapLoaders.map { case (featureShardId, indexMapLoader) =>

      val summary = FeatureDataStatistics(
        // Calling rdd explicitly here to avoid a typed encoder lookup in Spark 2.1
        data.select(featureShardId).rdd.map(_.getAs[SparkMLVector](0)),
        indexMapLoader.indexMapForDriver().get(Constants.INTERCEPT_KEY))

      (featureShardId, summary)
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

        val (_, baseConfig, evaluationResults) = models.head

        val iteration = getOrDefault(hyperParameterTuningIter)

        val dimension = baseConfig.toSeq.map {
          case (_, config: GLMOptimizationConfiguration) =>
            config.regularizationContext.regularizationType match {
              case RegularizationType.ELASTIC_NET => 2
              case RegularizationType.L2 => 1
              case RegularizationType.L1 => 1
              case RegularizationType.NONE => 0
            }
          case _ => throw new IllegalArgumentException(s"Unknown optimization config!")
        }.sum

        val mode = getOrDefault(hyperParameterTuning)

        val evaluator = evaluationResults.get.primaryEvaluator
        val isOptMax = evaluator.betterThan(1.0, 0.0)
        val evaluationFunction = new GameEstimatorEvaluationFunction(
          estimator,
          baseConfig,
          trainingData,
          testData,
          isOptMax)

        val observations = evaluationFunction.convertObservations(models)

        val hyperparameterTuner = HyperparameterTunerFactory[GameEstimator.GameResult](getOrDefault(hyperParameterTunerName))

        hyperparameterTuner.search(iteration, dimension, mode, evaluationFunction, observations)

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
   * @param models The models to evaluate (single evaluator, on the validation dataset)
   * @return The best model
   */
  protected[training] def selectBestModel(models: Seq[GameEstimator.GameResult]): GameEstimator.GameResult = {

    val bestResult = models
      .flatMap { case (model, modelConfig, evaluationsOpt) => evaluationsOpt.map((model, modelConfig, _)) }
      .reduceOption { (configModelEval1, configModelEval2) =>

        val (evaluations1, evaluations2) = (configModelEval1._3, configModelEval2._3)
        val evaluator1 = evaluations1.primaryEvaluator
        val evaluation1 = evaluations1.primaryEvaluation
        val evaluator2 = evaluations2.primaryEvaluator
        val evaluation2 = evaluations2.primaryEvaluation

        require(
          evaluator1 == evaluator2,
          "Evaluator mismatch while selecting best model; some error has occurred during validation.")

        if (evaluator1.betterThan(evaluation1, evaluation2)) configModelEval1 else configModelEval2
      }

    bestResult match {
      case Some(gameResult) =>
        val (model, configs, evaluations) = gameResult

        logger.info(s"Best model has ${evaluations.primaryEvaluator.name} score of ${evaluations.primaryEvaluation} " +
          s"and following config:")
        logger.info(IOUtils.optimizationConfigToString(configs))

        // Computing model summary is slow, we should only do it if necessary
        if (logger.isDebugEnabled) {
          logger.debug(s"Model summary:\n${model.toSummaryString}\n")
        }

        (model, configs, Some(evaluations))

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
        case Some((model, modelConfig, _)) =>

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
        case (modelIndex, (model, modelConfig, _)) =>

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
   * Write the per-group evaluation to HDFS.
   *
   * @param models All the models that were producing during training
   */
  private def savePerGroupEvaluationToHDFS(models: Seq[GameEstimator.GameResult]): Unit =

    if (getOrDefault(outputMode) != ModelOutputMode.NONE) {

      val hadoopConfiguration = sc.hadoopConfiguration
      val rootOutputDir = getRequiredParam(rootOutputDirectory)
      val allOutputDir = new Path(rootOutputDir, GROUP_EVAL_DIR)

      // Write additional models to HDFS
      models.foldLeft(0) {
        case (modelIndex, (_, _, evaluationOpt)) =>

          val evalOutputDir = new Path(allOutputDir, modelIndex.toString)

          Utils.createHDFSDir(evalOutputDir, hadoopConfiguration)
          IOUtils.saveGameEvaluationToHDFS(
            sparkSession,
            evalOutputDir,
            evaluationOpt,
            logger)

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

    sparkSession = SparkSessionConfiguration.asYarnClient(getOrDefault(applicationName), useKryo = true)
    sc = sparkSession.sparkContext
    logger = new PhotonLogger(new Path(getRequiredParam(rootOutputDirectory), LOGS_FILE_NAME), sc)
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
