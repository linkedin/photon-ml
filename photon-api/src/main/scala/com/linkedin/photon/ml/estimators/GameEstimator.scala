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
package com.linkedin.photon.ml.estimators

import scala.language.existentials

import org.apache.commons.cli.MissingArgumentException
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel
import org.slf4j.Logger

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types.{CoordinateId, FeatureShardId, UniqueSampleId}
import com.linkedin.photon.ml.algorithm._
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.evaluation.Evaluator.EvaluationResults
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.function.ObjectiveFunctionHelper
import com.linkedin.photon.ml.function.glm._
import com.linkedin.photon.ml.model.{GameModel, RandomEffectModel, RandomEffectModelInProjectedSpace}
import com.linkedin.photon.ml.normalization._
import com.linkedin.photon.ml.optimization.game._
import com.linkedin.photon.ml.projector.{IdentityProjection, IndexMapProjectorRDD, ProjectionMatrixBroadcast}
import com.linkedin.photon.ml.sampling.DownSamplerHelper
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.util._

/**
 * Estimator implementation for GAME models.
 *
 * @param sc The spark context for the application
 * @param logger The logger instance for the application
 */
class GameEstimator(val sc: SparkContext, implicit val logger: Logger) extends PhotonParams {

  import GameEstimator._

  // 2 types that make the code more readable
  type SingleNodeLossFunctionConstructor = (PointwiseLossFunction) => SingleNodeGLMLossFunction
  type DistributedLossFunctionConstructor = (PointwiseLossFunction) => DistributedGLMLossFunction

  private implicit val parent: Identifiable = this
  private val defaultNormalizationContext: NormalizationContextWrapper =
    NormalizationContextBroadcast(sc.broadcast(NoNormalization()))

  override val uid: String = Identifiable.randomUID(GAME_ESTIMATOR_PREFIX)

  //
  // Parameters
  //

  val trainingTask: Param[TaskType] = ParamUtils.createParam(
    "training task",
    "The type of training task to perform.")

  val inputColumnNames: Param[InputColumnsNames] = ParamUtils.createParam[InputColumnsNames](
    "input column names",
    "A map of custom column names which replace the default column names of expected fields in the Avro input.")

  val coordinateDataConfigurations: Param[Map[CoordinateId, CoordinateDataConfiguration]] =
    ParamUtils.createParam[Map[CoordinateId, CoordinateDataConfiguration]](
      "coordinate data configurations",
      "A map of coordinate names to data configurations.",
      PhotonParamValidators.nonEmpty[TraversableOnce, (CoordinateId, CoordinateDataConfiguration)])

  val coordinateUpdateSequence: Param[Seq[CoordinateId]] = ParamUtils.createParam(
    "coordinate update sequence",
    "The order in which coordinates are updated by the descent algorithm. It is recommended to order coordinates by " +
      "their stability (i.e. by looking at the variance of the feature distribution [or correlation with labels] for " +
      "each coordinate).",
    PhotonParamValidators.nonEmpty[Seq, CoordinateId])

  val coordinateDescentIterations: Param[Int] = ParamUtils.createParam(
    "coordinate descent iterations",
    "The number of coordinate descent iterations (one iteration is one full traversal of the update sequence).",
    ParamValidators.gt[Int](0.0))

  val coordinateNormalizationContexts: Param[Map[CoordinateId, NormalizationContext]] =
    ParamUtils.createParam[Map[CoordinateId, NormalizationContext]](
      "normalization contexts",
      "The normalization contexts for each coordinate. The type of normalization should be the same for each " +
        "coordinate, but the shifts and factors are different for each shard.",
      PhotonParamValidators.nonEmpty[TraversableOnce, (CoordinateId, NormalizationContext)])

  val initialModel: Param[GameModel] = ParamUtils.createParam(
    "initial model",
    "Prior model to use as a starting point for training.")

  val partialRetrainLockedCoordinates: Param[Set[CoordinateId]] = ParamUtils.createParam(
    "partial retrain locked coordinates",
    "The set of coordinates present in the pre-trained model to reuse during partial retraining.")

  val computeVariance: Param[Boolean] = ParamUtils.createParam[Boolean](
    "compute variance",
    "Whether to compute (approximate) coefficient variance.")

  val treeAggregateDepth: Param[Int] = ParamUtils.createParam[Int](
    "tree aggregate depth",
    "Suggested depth for tree aggregation.",
    ParamValidators.gt[Int](0.0))

  val validationEvaluators: Param[Seq[EvaluatorType]] = ParamUtils.createParam(
    "validation evaluators",
    "A list of evaluators used to validate computed scores (Note: the first evaluator in the list is the one used " +
      "for model selection)",
    PhotonParamValidators.nonEmpty[Seq, EvaluatorType])

  val ignoreThresholdForNewModels: Param[Boolean] = ParamUtils.createParam[Boolean](
    "ignore threshold for new models",
    "Flag to ignore the random effect samples lower bound when encountering a random effect ID without an existing " +
      "model during warm-start training.")

  //
  // Initialize object
  //

  setDefaultParams()

  //
  // Parameter setters
  //

  def setTrainingTask(value: TaskType): this.type = set(trainingTask, value)

  def setInputColumnNames(value: InputColumnsNames): this.type = set(inputColumnNames, value)

  def setCoordinateDataConfigurations(value: Map[CoordinateId, CoordinateDataConfiguration]): this.type =
    set(coordinateDataConfigurations, value)

  def setCoordinateUpdateSequence(value: Seq[CoordinateId]): this.type = set(coordinateUpdateSequence, value)

  def setCoordinateDescentIterations(value: Int): this.type = set(coordinateDescentIterations, value)

  def setCoordinateNormalizationContexts(value: Map[CoordinateId, NormalizationContext]): this.type =
    set(coordinateNormalizationContexts, value)

  def setInitialModel(value: GameModel): this.type = set(initialModel, value)

  def setPartialRetrainLockedCoordinates(value: Set[CoordinateId]): this.type =
    set(partialRetrainLockedCoordinates, value)

  def setComputeVariance(value: Boolean): this.type = set(computeVariance, value)

  def setTreeAggregateDepth(value: Int): this.type = set(treeAggregateDepth, value)

  def setValidationEvaluators(value: Seq[EvaluatorType]): this.type = set(validationEvaluators, value)

  def setIgnoreThresholdForNewModels(value: Boolean): this.type = set(ignoreThresholdForNewModels, value)

  //
  // Params trait extensions
  //

  override def copy(extra: ParamMap): GameEstimator = {

    val copy = new GameEstimator(sc, logger)

    extractParamMap(extra).toSeq.foreach { paramPair =>
      copy.set(copy.getParam(paramPair.param.name), paramPair.value)
    }

    copy
  }

  //
  // PhotonParams trait extensions
  //

  /**
   * Set the default parameters.
   */
  override protected def setDefaultParams(): Unit = {

    setDefault(inputColumnNames, InputColumnsNames())
    setDefault(coordinateDescentIterations, 1)
    setDefault(partialRetrainLockedCoordinates, Set.empty[CoordinateId])
    setDefault(computeVariance, false)
    setDefault(treeAggregateDepth, DEFAULT_TREE_AGGREGATE_DEPTH)
    setDefault(ignoreThresholdForNewModels, false)
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

    // Just need to check that the training task has been explicitly set
    getRequiredParam(trainingTask)

    val updateSequence = getRequiredParam(coordinateUpdateSequence)
    val dataConfigs = getRequiredParam(coordinateDataConfigurations)
    val initialModelOpt = get(initialModel)
    val retrainModelCoordsOpt = get(partialRetrainLockedCoordinates)
    val normalizationContextsOpt = get(coordinateNormalizationContexts)
    val ignoreThreshold = getOrDefault(ignoreThresholdForNewModels)
    val numUniqueCoordinates = updateSequence.toSet.size

    // Cannot have coordinates repeat in the update sequence
    require(
      numUniqueCoordinates == updateSequence.size,
      "One or more coordinates are repeated in the update sequence.")

    // Warm-start must be enabled to ignore threshold
    require(
      !ignoreThreshold || initialModelOpt.isDefined,
      "'Ignore threshold for new models' flag set but no initial model provided for warm-start")

    // Partial retraining and warm-start training require an initial GAME model to be provided as input
    val coordinatesToTrain = (initialModelOpt, retrainModelCoordsOpt) match {
      case (Some(initModel), Some(retrainModelCoords)) =>

        val newCoordinates = updateSequence.filterNot(retrainModelCoords.contains)

        // Locked coordinates cannot be empty
        require(
          retrainModelCoords.nonEmpty,
          "Set of locked coordinates is empty.")

        // No point in training if every coordinate is being reused
        require(
          newCoordinates.nonEmpty,
          "All coordinates in the update sequence are re-used from the initial model: no new coordinates to train.")

        // All locked coordinates must be used by the update sequence
        require(
          retrainModelCoords.forall(updateSequence.contains),
          "One or more locked coordinates for partial retraining are missing from the update sequence.")

        // All locked coordinates must be present in the initial model
        require(
          retrainModelCoords.forall(initModel.toMap.contains),
          "One or more locked coordinates for partial retraining are missing from the initial model.")

        newCoordinates

      case (Some(_), None) | (None, None) =>
        updateSequence

      case (None, Some(_)) =>
        throw new IllegalArgumentException("Partial retraining enabled, but no base model provided.")
    }

    // All coordinates (including locked coordinates) should have a data configuration
    updateSequence.foreach { coordinate =>
      require(
        dataConfigs.contains(coordinate),
        s"Coordinate $coordinate in the update sequence is missing data configuration.")
    }

    // If normalization is enabled, all non-locked coordinates must have a NormalizationContext
    coordinatesToTrain.foreach { coordinate =>
      require(
        normalizationContextsOpt.forall(normalizationContexts => normalizationContexts.contains(coordinate)),
        s"Coordinate $coordinate in the update sequence is missing normalization context")
    }
  }

  //
  // GameEstimator functions
  //

  /**
   * Fits a GAME model to the training dataset, once per configuration.
   *
   * @param data The training set
   * @param validationData Optional validation set for per-iteration validation
   * @param optimizationConfigurations A set of GAME optimization configurations
   * @return A set of (trained GAME model, optional evaluation results, GAME model configuration) tuples, one for each
   *         configuration
   */
  def fit(
      data: DataFrame,
      validationData: Option[DataFrame],
      optimizationConfigurations: Seq[GameOptimizationConfiguration]): Seq[GameResult] = {

    // Verify valid GameEstimator settings
    validateParams()

    // Verify valid function input
    validateInput(optimizationConfigurations)

    // Group additional columns to include in GameDatum
    val randomEffectIdCols: Set[String] = getRequiredParam(coordinateDataConfigurations)
      .flatMap { case (_, config) =>
        config match {
          case reConfig: RandomEffectDataConfiguration => Some(reConfig.randomEffectType)
          case _ => None
        }
      }
      .toSet
    val evaluatorCols = get(validationEvaluators).map(MultiEvaluatorType.getMultiEvaluatorIdTags).getOrElse(Set())
    val additionalCols = randomEffectIdCols ++ evaluatorCols

    // Transform the GAME dataset into fixed and random effect specific datasets
    val featureShards = getRequiredParam(coordinateDataConfigurations)
      .map { case (_, coordinateDataConfig) =>
        coordinateDataConfig.featureShardId
      }
      .toSet
    val (trainingDataSets, trainingLossFunctionEvaluator) = prepareTrainingDataSetsAndEvaluator(
      data,
      featureShards,
      additionalCols)
    val validationDataSetAndEvaluatorsOpt = prepareValidationDataSetAndEvaluators(
      validationData,
      featureShards,
      additionalCols)
    val normalizationContextWrappersOpt = prepareNormalizationContextWrappers(trainingDataSets)
    val coordinateDescent = new CoordinateDescent(
      getRequiredParam(coordinateUpdateSequence),
      getOrDefault(coordinateDescentIterations),
      trainingLossFunctionEvaluator,
      validationDataSetAndEvaluatorsOpt,
      getOrDefault(partialRetrainLockedCoordinates),
      logger)

    val results = Timed(s"Training models:") {

      var prevGameModel: Option[GameModel] = getInitialModel(trainingDataSets)

      optimizationConfigurations.map { optimizationConfiguration =>
        val (gameModel, evaluation) = train(
          optimizationConfiguration,
          trainingDataSets,
          coordinateDescent,
          normalizationContextWrappersOpt,
          prevGameModel)

        prevGameModel = Some(gameModel)

        (gameModel, evaluation, optimizationConfiguration)
      }
    }

    // Purge the training data, validation data, and normalization contexts
    trainingDataSets.foreach { case (_, dataSet) =>
      dataSet match {
        case rddLike: RDDLike => rddLike.unpersistRDD()
        case _ =>
      }
      dataSet match {
        case broadcastLike: BroadcastLike => broadcastLike.unpersistBroadcast()
        case _ =>
      }
    }
    validationDataSetAndEvaluatorsOpt.map { case (rdd, _) => rdd.unpersist() }
    normalizationContextWrappersOpt.foreach(_.foreach { case (_, nCW) => nCW.unpersist() } )

    results
  }

  /**
   * Gets and initializes the prior model, if specified
   *
   * @param trainingDataSets Per-item prior models are projected with the projectors of the current dataset, so
   *   initializing the model is a function of the current dataset.
   * @return The initial model
   */
  protected def getInitialModel(
      trainingDataSets: Map[CoordinateId, D forSome { type D <: DataSet[D] }]): Option[GameModel] =
    get(initialModel).map { gameModel =>
      new GameModel(gameModel
        .toMap
        .map { case (coordinateId, model) => (model, trainingDataSets(coordinateId)) match {
          // For random effect models, we need to transform the prior model into the projected space of the current
          // dataset
          case (reModel: RandomEffectModel, reDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace) =>
            (coordinateId,
              new RandomEffectModelInProjectedSpace(
                reDataSetInProjectedSpace.randomEffectProjector.transformCoefficientsRDD(reModel.modelsRDD),
                reDataSetInProjectedSpace.randomEffectProjector,
                reModel.randomEffectType,
                reModel.featureShardId).persistRDD(StorageLevel.DISK_ONLY))

          case _ =>
            (coordinateId, model)
        }})
    }

  /**
   * Verify that the input to which we're fitting a model is valid.
   *
   * @param optimizationConfigurations A set of GAME optimization configurations
   */
  protected def validateInput(optimizationConfigurations: Seq[GameOptimizationConfiguration]): Unit = {

    val updateSequence = getRequiredParam(coordinateUpdateSequence)
    val lockedCoordinatesOpt = get(partialRetrainLockedCoordinates)
    val coordinatesToTrain = lockedCoordinatesOpt
      .map(lockedCoordinates => updateSequence.filterNot(lockedCoordinates.contains))
      .getOrElse(updateSequence)

    optimizationConfigurations.foreach { gameOptimizationConfig =>

      val gameOptConfigCoords = gameOptimizationConfig.keySet
      val missingCoords = coordinatesToTrain.filterNot(gameOptConfigCoords.contains)
      val badLockedCoords = gameOptConfigCoords.intersect(lockedCoordinatesOpt.getOrElse(Set()))

      // New or retrained coordinates from the update sequence must be present in the optimization configurations
      require(
        missingCoords.isEmpty,
        s"Coordinates '${missingCoords.mkString(", ")}' required by the update sequence are missing from the " +
          s"optimization configurations.")
      // Locked coordinates should not be present in the optimization configurations
      require(
        badLockedCoords.isEmpty,
        s"Locked coordinates for partial retraining '${badLockedCoords.mkString(", ")}' are present in the " +
          s"optimization configurations."
      )

      // TODO: Should coordinates not used by the update sequence be banned from the optimization configurations?
    }
  }

  /**
   * Construct one or more training [[DataSet]]s using a [[DataFrame]] of training data and an [[Evaluator]] for
   * computing training loss.
   *
   * @param data The training data [[DataFrame]]
   * @param idTagSet A set of additional columns whose values should be maintained for training
   * @return A (map of training [[DataSet]]s (one per coordinate), training loss [[Evaluator]]) tuple
   */
  protected def prepareTrainingDataSetsAndEvaluator(
      data: DataFrame,
      featureShards: Set[FeatureShardId],
      idTagSet: Set[String]): (Map[CoordinateId, D forSome { type D <: DataSet[D] }], Evaluator) = {

    val numPartitions = data.rdd.getNumPartitions
    val gameDataPartitioner = new LongHashPartitioner(numPartitions)

    val gameDataSet = Timed("Process training data from raw dataframe to RDD of samples") {
      GameConverters
        .getGameDataSetFromDataFrame(
          data,
          featureShards,
          idTagSet,
          isResponseRequired = true,
          getOrDefault(inputColumnNames))
        .partitionBy(gameDataPartitioner)
        .setName("GAME training data")
        .persist(StorageLevel.DISK_ONLY)
    }
    // Transform the GAME dataset into fixed and random effect specific datasets
    val trainingDataSet = Timed("Prepare training datasets") {
      prepareTrainingDataSets(gameDataSet)
    }
    val trainingLossFunctionEvaluator = Timed("Prepare training loss evaluator") {
      prepareTrainingLossEvaluator(gameDataSet)
    }

    // Purge the GAME dataset, which is no longer needed in the following code
    gameDataSet.unpersist()

    (trainingDataSet, trainingLossFunctionEvaluator)
  }

  /**
   * Construct one or more [[DataSet]]s from an [[RDD]] of samples.
   *
   * @param gameDataSet The training data samples
   * @return A map of coordinate ID to training [[DataSet]]
   */
  protected def prepareTrainingDataSets(
      gameDataSet: RDD[(UniqueSampleId, GameDatum)]): Map[CoordinateId, D forSome { type D <: DataSet[D] }] = {

    getRequiredParam(coordinateDataConfigurations).map { case (coordinateId, config) =>

      val result = config match {

        case feConfig: FixedEffectDataConfiguration =>

          val fixedEffectDataSet = FixedEffectDataSet(gameDataSet, feConfig.featureShardId)
            .setName(s"Fixed Effect Data Set: $coordinateId")
            .persistRDD(StorageLevel.DISK_ONLY)

          if (logger.isDebugEnabled) {
            // Eval this only in debug mode, because the call to "toSummaryString" can be very expensive
            logger.debug(
              s"Summary of fixed effect dataset with coordinate ID '$coordinateId':\n" +
                s"${fixedEffectDataSet.toSummaryString}")
          }

          (coordinateId, fixedEffectDataSet)

        case reConfig: RandomEffectDataConfiguration =>

          val rePartitioner = RandomEffectDataSetPartitioner.fromGameDataSet(gameDataSet, reConfig)

          val existingModelKeysRddOpt = if (getOrDefault(ignoreThresholdForNewModels)) {
            getRequiredParam(initialModel).getModel(coordinateId).map {
              case rem: RandomEffectModel =>
                rem.modelsRDD.partitionBy(rePartitioner).keys

              case other =>
                throw new IllegalArgumentException(
                  s"Model type mismatch: expected Random Effect Model but found '${other.getClass}'")
            }
          } else {
            None
          }

          val rawRandomEffectDataSet = RandomEffectDataSet(gameDataSet, reConfig, rePartitioner, existingModelKeysRddOpt)
            .setName(s"Random Effect Data Set: $coordinateId")
            .persistRDD(StorageLevel.DISK_ONLY)
            .materialize()
          val projectorType = reConfig.projectorType
          val randomEffectDataSet = projectorType match {

            case IdentityProjection => rawRandomEffectDataSet

            case _ =>

              val randomEffectDataSetInProjectedSpace = RandomEffectDataSetInProjectedSpace
                .buildWithProjectorType(rawRandomEffectDataSet, projectorType)
                .setName(s"Projected Random Effect Data Set: $coordinateId")
                .persistRDD(StorageLevel.DISK_ONLY)
                .materialize()

              // Only un-persist the active data and passive data, because randomEffectDataSet and
              // randomEffectDataSetInProjectedSpace share uniqueIdToRandomEffectIds and other RDDs/Broadcasts.
              //
              // Do not un-persist for identity projection.
              projectorType match {
                case IdentityProjection =>

                case _ =>
                  rawRandomEffectDataSet.activeData.unpersist()
                  rawRandomEffectDataSet.passiveDataOption.foreach(_.unpersist())
              }

              randomEffectDataSetInProjectedSpace
          }

          if (logger.isDebugEnabled) {
            // Eval this only in debug mode, because the call to "toSummaryString" can be very expensive
            logger.debug(
              s"Summary of random effect dataset with coordinate ID $coordinateId:\n" +
                s"${randomEffectDataSet.toSummaryString}\n")
          }

          rePartitioner.unpersistBroadcast()

          (coordinateId, randomEffectDataSet)
      }

      result.asInstanceOf[(CoordinateId, D forSome { type D <: DataSet[D] })]
    }
  }

  /**
   * Construct the training loss evaluator.
   *
   * @param gameDataSet The training data samples
   * @return A training loss evaluator for the given training task and data
   */
  protected def prepareTrainingLossEvaluator(gameDataSet: RDD[(UniqueSampleId, GameDatum)]): Evaluator = {

    val taskType = getRequiredParam(trainingTask)
    val labelAndOffsetAndWeights = gameDataSet
      .mapValues(gameData => (gameData.response, gameData.offset, gameData.weight))
      .setName("Training labels, offsets and weights")
      .persist(StorageLevel.MEMORY_AND_DISK)

    labelAndOffsetAndWeights.count()

    taskType match {
      case TaskType.LOGISTIC_REGRESSION =>
        new LogisticLossEvaluator(labelAndOffsetAndWeights)
      case TaskType.LINEAR_REGRESSION =>
        new SquaredLossEvaluator(labelAndOffsetAndWeights)
      case TaskType.POISSON_REGRESSION =>
        new PoissonLossEvaluator(labelAndOffsetAndWeights)
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        new SmoothedHingeLossEvaluator(labelAndOffsetAndWeights)
      case _ =>
        throw new UnsupportedOperationException(s"$taskType is not a valid GAME training task")
    }
  }

  /**
   * Optionally construct an [[RDD]] of validation data samples using a [[DataFrame]] of validation data and the
   * validation metric [[Evaluator]]s.
   *
   * @param dataOpt Optional validation data [[DataFrame]]
   * @param additionalCols A set of additional columns whose values should be maintained for validation evaluation
   * @return An optional ([[RDD]] of validation samples, validation metric [[Evaluator]]s) tuple
   */
  protected def prepareValidationDataSetAndEvaluators(
      dataOpt: Option[DataFrame],
      featureShards: Set[FeatureShardId],
      additionalCols: Set[String]): Option[(RDD[(UniqueSampleId, GameDatum)], Seq[Evaluator])] =

    dataOpt.map { data =>
      val partitioner = new LongHashPartitioner(data.rdd.partitions.length)
      val gameDataSet = Timed("Convert training data from raw dataframe to processed RDD") {
        val result = GameConverters
          .getGameDataSetFromDataFrame(
            data,
            featureShards,
            additionalCols,
            isResponseRequired = true,
            getOrDefault(inputColumnNames))
          .partitionBy(partitioner)
          .setName("Validating Game dataset")
          .persist(StorageLevel.DISK_ONLY)

        result.count()
        result
      }

      val evaluators = Timed("Prepare validation metric evaluators") {
        prepareValidationEvaluators(gameDataSet)
      }

      val randomScores = gameDataSet.mapValues(_ => math.random)
      evaluators.foreach { evaluator =>
        val metric = evaluator.evaluate(randomScores)
        logger.info(s"Random guessing based baseline evaluation metric for ${evaluator.getEvaluatorName}: $metric")
      }

      (gameDataSet, evaluators)
    }

  /**
   * Construct the validation [[Evaluator]]s.
   *
   * @param gameDataSet An [[RDD]] of validation data samples
   * @return One or more validation metric [[Evaluator]]s
   */
  protected def prepareValidationEvaluators(gameDataSet: RDD[(UniqueSampleId, GameDatum)]): Seq[Evaluator] = {

    val validatingLabelsAndOffsetsAndWeights = gameDataSet
      .mapValues(gameData => (gameData.response, gameData.offset, gameData.weight))
      .setName(s"Validating labels and offsets")
      .persist(StorageLevel.MEMORY_AND_DISK)
    validatingLabelsAndOffsetsAndWeights.count()

    get(validationEvaluators)
      .map(_.map(EvaluatorFactory.buildEvaluator(_, gameDataSet)))
      .getOrElse {
        // Get default evaluators given the task type
        val defaultEvaluator =
          getRequiredParam(trainingTask) match {
            case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
              new AreaUnderROCCurveEvaluator(validatingLabelsAndOffsetsAndWeights)

            case TaskType.LINEAR_REGRESSION =>
              new RMSEEvaluator(validatingLabelsAndOffsetsAndWeights)

            case TaskType.POISSON_REGRESSION =>
              new PoissonLossEvaluator(validatingLabelsAndOffsetsAndWeights)

            case _ =>
              throw new UnsupportedOperationException(
                s"${getRequiredParam(trainingTask)} is not a valid GAME training task")
          }

        Seq(defaultEvaluator)
      }
  }

  protected def prepareNormalizationContextWrappers(
      dataSets: Map[CoordinateId, D forSome { type D <: DataSet[D] }])
    : Option[Map[CoordinateId, NormalizationContextWrapper]] =

    get(coordinateNormalizationContexts).map { normalizationContextsMap =>

      normalizationContextsMap.map { case (coordinate, normalizationContext) =>

        val normalization = dataSets(coordinate) match {

          case reInProjSpace: RandomEffectDataSetInProjectedSpace =>
            reInProjSpace.randomEffectProjector match {

              case indexProj: IndexMapProjectorRDD =>
                NormalizationContextRDD(indexProj.projectNormalizationRDD(normalizationContext))

              case randomProj: ProjectionMatrixBroadcast =>
                NormalizationContextBroadcast(sc.broadcast(randomProj.projectNormalizationContext(normalizationContext)))

              case _ =>
                NormalizationContextBroadcast(sc.broadcast(normalizationContext))
            }

          case _ =>
            NormalizationContextBroadcast(sc.broadcast(normalizationContext))
        }

        (coordinate, normalization)
      }
    }

  /**
   * Fit a GAME model with the given configuration to the given training data.
   *
   * The configuration is used to define one or more 'coordinates' of various types (i.e. fixed effect, random effect,
   * matrix factorization) which together for the complete mixed effect optimization problem. The coordinates are
   * updated in a given order, a given number of times. For optimum performance, the update order should start
   * with the most general 'coordinates' and end with the least general - each successive update learning the residuals
   * of the previous 'coordinates'.
   *
   * @param configuration The configuration for the GAME optimization problem
   * @param trainingDataSets The training datasets for each coordinate of the GAME optimization problem
   * @param coordinateDescent The coordinate descent driver
   * @param normalizationContextWrappersOpt Optional normalization contexts, wrapped for use by random effect
   *                                        coordinates
   * @param initialModelOpt An optional existing GAME model who's components should be used to warm-start training
   * @return A trained GAME model
   */
  protected def train(
      configuration: GameOptimizationConfiguration,
      trainingDataSets: Map[CoordinateId, D forSome { type D <: DataSet[D] }],
      coordinateDescent: CoordinateDescent,
      normalizationContextWrappersOpt: Option[Map[CoordinateId, NormalizationContextWrapper]],
      initialModelOpt: Option[GameModel] = None): (GameModel, Option[EvaluationResults]) = Timed(s"Train model:") {

    logger.info("Model configuration:")
    configuration.foreach { case (coordinateId, coordinateConfig) =>
      logger.info(s"coordinate '$coordinateId':\n$coordinateConfig")
    }

    val task = getRequiredParam(trainingTask)
    val updateSequence = getRequiredParam(coordinateUpdateSequence)
    val variance = getOrDefault(computeVariance)
    val lossFunctionFactory = ObjectiveFunctionHelper.buildFactory(task, getOrDefault(treeAggregateDepth))
    val glmConstructor = task match {
      case TaskType.LOGISTIC_REGRESSION => LogisticRegressionModel.apply _
      case TaskType.LINEAR_REGRESSION => LinearRegressionModel.apply _
      case TaskType.POISSON_REGRESSION => PoissonRegressionModel.apply _
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => SmoothedHingeLossLinearSVMModel.apply _
      case _ => throw new Exception("Need to specify a valid loss function")
    }
    val downSamplerFactory = DownSamplerHelper.buildFactory(task)
    val normalizationContextsWrappers = normalizationContextWrappersOpt
      .getOrElse(configuration.mapValues(_ => defaultNormalizationContext))
    val lockedCoordinates = get(partialRetrainLockedCoordinates).getOrElse(Set())

    // Create the optimization coordinates for each component model
    val coordinates: Map[CoordinateId, C forSome { type C <: Coordinate[_] }] =
      updateSequence
        .map { coordinateId =>
          val coordinate: C forSome { type C <: Coordinate[_] } = if (lockedCoordinates.contains(coordinateId)) {
            trainingDataSets(coordinateId) match {
              case feDataSet: FixedEffectDataSet => new FixedEffectModelCoordinate(feDataSet)
              case reDataSet: RandomEffectDataSet => new RandomEffectModelCoordinate(reDataSet)
              case dataSet => throw new UnsupportedOperationException(s"Unsupported dataset type: ${dataSet.getClass}")
            }
          } else {
            CoordinateFactory.build(
              trainingDataSets(coordinateId),
              configuration(coordinateId),
              lossFunctionFactory,
              glmConstructor,
              downSamplerFactory,
              normalizationContextsWrappers(coordinateId),
              TRACK_STATE,
              variance)
          }

          (coordinateId, coordinate)
        }
        .toMap

    // Create the base model to optimize, using a combination of existing coordinates (if provided) and new ones
    val model = updateSequence
      .map { coordinateId =>

        val initializedModel = initialModelOpt
          .flatMap(_.getModel(coordinateId))
          .getOrElse(coordinates(coordinateId).initializeModel(MathConst.RANDOM_SEED))

        initializedModel match {
          case rddLike: RDDLike =>
            rddLike
              .setName(s"Initialized model with coordinate id $coordinateId")
              .persistRDD(StorageLevel.DISK_ONLY)

          case _ =>
        }

        if (logger.isDebugEnabled) {
          logger.debug(
            s"Summary of model (${initializedModel.getClass}) initialized for coordinate with ID $coordinateId:" +
              s"\n${initializedModel.toSummaryString}")
        }

        (coordinateId, initializedModel)
      }
      .toMap

    coordinateDescent.run(coordinates, new GameModel(model))
  }
}

object GameEstimator {

  //
  // Types
  //

  type GameOptimizationConfiguration = Map[CoordinateId, CoordinateOptimizationConfiguration]
  type GameResult = (GameModel, Option[EvaluationResults], GameOptimizationConfiguration)

  //
  // Constants
  //

  private val GAME_ESTIMATOR_PREFIX = "GameEstimator"

  val DEFAULT_TREE_AGGREGATE_DEPTH = 1
  val TRACK_STATE = true
}
