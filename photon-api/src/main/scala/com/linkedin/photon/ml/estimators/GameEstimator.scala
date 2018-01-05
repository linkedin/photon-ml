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

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.slf4j.Logger

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types.{CoordinateId, FeatureShardId, UniqueSampleId}
import com.linkedin.photon.ml.algorithm._
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.evaluation.Evaluator.EvaluationResults
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.function.glm._
import com.linkedin.photon.ml.function.svm.{DistributedSmoothedHingeLossFunction, SingleNodeSmoothedHingeLossFunction}
import com.linkedin.photon.ml.function.{DistributedObjectiveFunction, SingleNodeObjectiveFunction}
import com.linkedin.photon.ml.model.GameModel
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game._
import com.linkedin.photon.ml.projector.IdentityProjection
import com.linkedin.photon.ml.sampler.{BinaryClassificationDownSampler, DefaultDownSampler, DownSampler}
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.util.Implicits._
import com.linkedin.photon.ml.util._

/**
 * Estimator implementation for GAME models.
 *
 * @param sc The spark context for the application
 * @param logger The logger instance for the application
 */
class GameEstimator(val sc: SparkContext, implicit val logger: Logger) extends Params {

  import GameEstimator._

  // 2 types that make the code more readable
  type SingleNodeLossFunctionConstructor = (PointwiseLossFunction) => SingleNodeGLMLossFunction
  type DistributedLossFunctionConstructor = (PointwiseLossFunction) => DistributedGLMLossFunction

  private implicit val parent: Identifiable = this
  private val defaultNormalizationContext: BroadcastWrapper[NormalizationContext] = PhotonBroadcast(sc.broadcast(NoNormalization()))

  override val uid: String = Identifiable.randomUID(GAME_ESTIMATOR_PREFIX)

  //
  // Parameters
  //

  val trainingTask: Param[TaskType] = ParamUtils.createParam(
    "training task",
    "The type of training task to perform.",
    {taskType: TaskType => taskType != TaskType.NONE})

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

  val coordinateNormalizationContexts: Param[Map[CoordinateId, BroadcastWrapper[NormalizationContext]]] =
    ParamUtils.createParam[Map[CoordinateId, BroadcastWrapper[NormalizationContext]]](
      "normalization contexts",
      "The normalization contexts for each coordinate. The type of normalization should be the same for each " +
        "coordinate, but the shifts and factors are different for each shard.",
      PhotonParamValidators.nonEmpty[TraversableOnce, (CoordinateId, BroadcastWrapper[NormalizationContext])])

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

  val useWarmStart: Param[Boolean] = ParamUtils.createParam[Boolean](
    "use warm start",
    "Whether to re-use trained GAME models as starting points.")

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

  def setCoordinateNormalizationContexts(value: Map[CoordinateId, BroadcastWrapper[NormalizationContext]]): this.type =
    set(coordinateNormalizationContexts, value)

  def setComputeVariance(value: Boolean): this.type = set(computeVariance, value)

  def setTreeAggregateDepth(value: Int): this.type = set(treeAggregateDepth, value)

  def setValidationEvaluators(value: Seq[EvaluatorType]): this.type = set(validationEvaluators, value)

  def setWarmStart(value: Boolean): this.type = set(useWarmStart, value)

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

  /**
   * Set the default parameters.
   */
  private def setDefaultParams(): Unit = {

    setDefault(coordinateDescentIterations, 1)
    setDefault(inputColumnNames, InputColumnsNames())
    setDefault(computeVariance, false)
    setDefault(treeAggregateDepth, DEFAULT_TREE_AGGREGATE_DEPTH)
    setDefault(useWarmStart, true)
  }

  /**
   * Verify that the interactions between individual parameters are valid.
   *
   * @note In Spark, interactions between parameters are checked by
   *       [[org.apache.spark.ml.PipelineStage.transformSchema()]]. Since we do not use the Spark pipeline API in
   *       Photon-ML, we need to have this function to check the interactions between parameters.
   *
   * @throws IllegalArgumentException if a required parameter is missing or a validation check fails
   */
  protected[estimators] def validateParams(): Unit = {

    // Just need to check that the training task has been explicitly set
    getRequiredParam(trainingTask)

    val updateSequence = getRequiredParam(coordinateUpdateSequence)
    val dataConfigs = getRequiredParam(coordinateDataConfigurations)
    val normalizationContextsOpt = get(coordinateNormalizationContexts)
    val numUniqueCoordinates = updateSequence.toSet.size

    require(
      numUniqueCoordinates == updateSequence.size,
      "Mismatch between coordinate configurations and update sequence.")

    updateSequence.foreach { coordinate =>
      require(
        dataConfigs.contains(coordinate),
        s"Coordinate $coordinate in the update sequence is missing data configuration.")
      require(
        normalizationContextsOpt.forall(normalizationContexts => normalizationContexts.contains(coordinate)),
        s"Coordinate $coordinate in the update sequence is missing normalization context")
    }
  }

  /**
   * Return the user-supplied value for a required parameter. Used for mandatory parameters without default values.
   *
   * @tparam T The type of the parameter
   * @param param The parameter
   * @return The value associated with the parameter
   * @throws IllegalArgumentException if no value is associated with the given parameter
   */
  private def getRequiredParam[T](param: Param[T]): T =
    get(param)
      .getOrElse(throw new IllegalArgumentException(s"Missing required parameter ${param.name}"))

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
    val validationDataSetAndEvaluators = prepareValidationDataSetAndEvaluators(
      validationData,
      featureShards,
      additionalCols)

    val results = Timed(s"Training models:") {

      var prevGameModel: Option[GameModel] = None

      optimizationConfigurations.map { optimizationConfiguration =>
        val (gameModel, evaluation) = train(
          optimizationConfiguration,
          trainingDataSets,
          trainingLossFunctionEvaluator,
          validationDataSetAndEvaluators,
          prevGameModel)

        if (getOrDefault(useWarmStart)) {
          prevGameModel = Some(gameModel)
        }

        (gameModel, evaluation, optimizationConfiguration)
      }
    }

    // Purge the training set
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

    results
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
        .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    }
    // Transform the GAME dataset into fixed and random effect specific datasets
    val trainingDataSet = Timed("Prepare training data sets") {
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
            .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

          logger.debug(
            s"Summary of fixed effect data set with coordinate ID '$coordinateId':\n" +
              s"${fixedEffectDataSet.toSummaryString}")

          (coordinateId, fixedEffectDataSet)

        case reConfig: RandomEffectDataConfiguration =>

          val partitioner = RandomEffectDataSetPartitioner.fromGameDataSet(
            reConfig.minNumPartitions,
            reConfig.randomEffectType,
            gameDataSet)
          val rawRandomEffectDataSet = RandomEffectDataSet(gameDataSet, reConfig, partitioner)
            .setName(s"Random Effect Data Set: $coordinateId")
            .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
            .materialize()
          val projectorType = reConfig.projectorType
          val randomEffectDataSet = projectorType match {

            case IdentityProjection => rawRandomEffectDataSet

            case _ =>

              val randomEffectDataSetInProjectedSpace = RandomEffectDataSetInProjectedSpace
                .buildWithProjectorType(rawRandomEffectDataSet, projectorType)
                .setName(s"Projected Random Effect Data Set: $coordinateId")
                .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
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

          logger.debug(
            s"Summary of random effect data set with coordinate ID $coordinateId:\n" +
              s"${randomEffectDataSet.toSummaryString}\n")

          partitioner.unpersistBroadcast()

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
      .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)

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
          .setName("Validating Game data set")
          .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

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
      .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
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
   * @param trainingDataSets The training data sets for each coordinate of the GAME optimization problem
   * @param trainingEvaluator The evaluator for the training function loss
   * @param validationDataAndEvaluators Optional validation data and evaluators
   * @return A trained GAME model
   */
  protected def train(
      configuration: GameOptimizationConfiguration,
      trainingDataSets: Map[CoordinateId, D forSome { type D <: DataSet[D] }],
      trainingEvaluator: Evaluator,
      validationDataAndEvaluators: Option[(RDD[(Long, GameDatum)], Seq[Evaluator])],
      prevGameModelOpt: Option[GameModel] = None)
    : (GameModel, Option[EvaluationResults]) = Timed(s"Train model:") {

    logger.info("Model configuration:")
    configuration.foreach { case (coordinateId, coordinateConfig) =>
      logger.info(s"coordinate '$coordinateId':\n$coordinateConfig")
    }

    val normalizationContexts = get(coordinateNormalizationContexts)
    val variance = getOrDefault(computeVariance)
    val glmConstructor = getRequiredParam(trainingTask) match {
      case TaskType.LOGISTIC_REGRESSION => LogisticRegressionModel.apply _
      case TaskType.LINEAR_REGRESSION => LinearRegressionModel.apply _
      case TaskType.POISSON_REGRESSION => PoissonRegressionModel.apply _
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => SmoothedHingeLossLinearSVMModel.apply _
      case _ => throw new Exception("Need to specify a valid loss function")
    }

    // For each model, create the optimization coordinates
    val coordinates = getRequiredParam(coordinateUpdateSequence).map { coordinateId =>
      val coordinate: Coordinate[_] = (configuration(coordinateId), trainingDataSets(coordinateId)) match {

        case (feOptConfig: FixedEffectOptimizationConfiguration, feDataSet: FixedEffectDataSet) =>
          // If number of features is from moderate to large (>200000), then use a deeper tree aggregate
          val treeAggDepth = if (feDataSet.numFeatures < FIXED_EFFECT_FEATURE_THRESHOLD) {
            getOrDefault(treeAggregateDepth)
          } else {
            Math.max(getOrDefault(treeAggregateDepth), DEEP_TREE_AGGREGATE_DEPTH)
          }

          new FixedEffectCoordinate(
            feDataSet,
            DistributedOptimizationProblem(
              feOptConfig,
              selectDistributedLossFunction(feOptConfig, treeAggDepth),
              setupDownSampler(feOptConfig.downSamplingRate),
              glmConstructor,
              normalizationContexts.extractOrElse(coordinateId)(defaultNormalizationContext),
              TRACK_STATE,
              variance))

        case (reOptConfig: RandomEffectOptimizationConfiguration, reDataSet: RandomEffectDataSetInProjectedSpace) =>
          new RandomEffectCoordinateInProjectedSpace(
            reDataSet,
            RandomEffectOptimizationProblem(
                reDataSet,
                reOptConfig,
                selectSingleNodeLossFunction(reOptConfig),
                glmConstructor,
                normalizationContexts.extractOrElse(coordinateId)(defaultNormalizationContext),
                TRACK_STATE,
                variance)
              .setName(s"Random effect optimization problem of coordinate $coordinateId")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL))

        case (optimizationConfig, dataSet) =>
          throw new UnsupportedOperationException(
            s"Unsupported (configuration, data set) pair: (${optimizationConfig.getClass}, ${dataSet.getClass})")
      }

      Pair[CoordinateId, Coordinate[_]](coordinateId, coordinate)
    }

    prevGameModelOpt match {
      case Some(prevGameModel) =>
        CoordinateDescent(coordinates, trainingEvaluator, validationDataAndEvaluators, logger)
          .run(getOrDefault(coordinateDescentIterations), prevGameModel)

      case None =>
        CoordinateDescent(coordinates, trainingEvaluator, validationDataAndEvaluators, logger)
          .run(getOrDefault(coordinateDescentIterations))
    }
  }

  /**
   * Sets the sampling option for this model training run.
   *
   * @param downSamplingRate the downSamplingRate
   * @return An Option containing a DownSampler or None
   */
  protected[estimators] def setupDownSampler(downSamplingRate: Double): Option[DownSampler] =

    if (downSamplingRate > 0D && downSamplingRate < 1D) {
      getRequiredParam(trainingTask) match {
        case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
          Some(new BinaryClassificationDownSampler(downSamplingRate))

        case TaskType.LINEAR_REGRESSION | TaskType.POISSON_REGRESSION =>
          Some(new DefaultDownSampler(downSamplingRate))
      }
    } else {
      None
    }

  /**
   * Construct the functions used for computing loss by a factored random effect optimization problem.
   *
   * @param randomEffectOptimizationConfiguration The random effect optimization problem configuration
   * @param latentFactorOptimizationConfiguration The latent factor optimization problem configuration
   * @return The loss functions used by factored random effect optimization
   */
  private def selectRandomLatentObjectiveFunction(
      randomEffectOptimizationConfiguration: GLMOptimizationConfiguration,
      latentFactorOptimizationConfiguration: GLMOptimizationConfiguration)
    : (SingleNodeObjectiveFunction, DistributedObjectiveFunction) = {

    val lossFunction: SingleNodeLossFunctionConstructor = SingleNodeGLMLossFunction(
      randomEffectOptimizationConfiguration)
    val distLossFunction: DistributedLossFunctionConstructor = DistributedGLMLossFunction(
      sc,
      latentFactorOptimizationConfiguration,
      getOrDefault(treeAggregateDepth))

    getRequiredParam(trainingTask) match {
      case TaskType.LOGISTIC_REGRESSION => (lossFunction(LogisticLossFunction), distLossFunction(LogisticLossFunction))
      case TaskType.LINEAR_REGRESSION => (lossFunction(SquaredLossFunction), distLossFunction(SquaredLossFunction))
      case TaskType.POISSON_REGRESSION => (lossFunction(PoissonLossFunction), distLossFunction(PoissonLossFunction))
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        (SingleNodeSmoothedHingeLossFunction(randomEffectOptimizationConfiguration),
          DistributedSmoothedHingeLossFunction(
            sc,
            latentFactorOptimizationConfiguration,
            getOrDefault(treeAggregateDepth)))
    }
  }

  /**
   * Construct a function for computing loss on a single node (used by random effect optimization problems).
   *
   * @param configuration An optimization configuration
   * @return The loss function
   */
  private def selectSingleNodeLossFunction(configuration: GLMOptimizationConfiguration): SingleNodeObjectiveFunction = {

    val lossFunction: SingleNodeLossFunctionConstructor = SingleNodeGLMLossFunction(configuration)

    getRequiredParam(trainingTask) match {
      case TaskType.LOGISTIC_REGRESSION => lossFunction(LogisticLossFunction)
      case TaskType.LINEAR_REGRESSION => lossFunction(SquaredLossFunction)
      case TaskType.POISSON_REGRESSION => lossFunction(PoissonLossFunction)
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => SingleNodeSmoothedHingeLossFunction(configuration)
    }
  }

  /**
   * Construct a function for computing loss on one or more nodes (used by fixed effect optimization problems).
   *
   * @param configuration An optimization configuration
   * @param treeAggregateDepth The depth to use during Spark tree aggregation (1 = regular aggregation)
   * @return The loss function
   */
  private def selectDistributedLossFunction(
      configuration: GLMOptimizationConfiguration,
      treeAggregateDepth: Int): DistributedObjectiveFunction = {

    val distLossFunction: DistributedLossFunctionConstructor = DistributedGLMLossFunction(
      sc,
      configuration,
      treeAggregateDepth)

    getRequiredParam(trainingTask) match {
      case TaskType.LOGISTIC_REGRESSION => distLossFunction(LogisticLossFunction)
      case TaskType.LINEAR_REGRESSION => distLossFunction(SquaredLossFunction)
      case TaskType.POISSON_REGRESSION => distLossFunction(PoissonLossFunction)
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        DistributedSmoothedHingeLossFunction(sc, configuration, treeAggregateDepth)
    }
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

  val FIXED_EFFECT_FEATURE_THRESHOLD = 200000
  val DEFAULT_TREE_AGGREGATE_DEPTH = 1
  val DEEP_TREE_AGGREGATE_DEPTH = 2
  val TRACK_STATE = true
}
