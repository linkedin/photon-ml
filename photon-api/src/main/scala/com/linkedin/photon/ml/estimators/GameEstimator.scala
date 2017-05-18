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

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.slf4j.Logger

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types.{UniqueSampleId, FeatureShardId, CoordinateId}
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
import com.linkedin.photon.ml.optimization.DistributedOptimizationProblem
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
class GameEstimator(val sc: SparkContext, implicit val logger: Logger) {

  import GameEstimator._

  // 2 types that makes the code more readable
  type SingleNodeLossFunctionConstructor = (PointwiseLossFunction) => SingleNodeGLMLossFunction
  type DistributedLossFunctionConstructor = (PointwiseLossFunction) => DistributedGLMLossFunction

  private val defaultNormalizationContext: Broadcast[NormalizationContext] = sc.broadcast(NoNormalization())

  /**
   * Column names of the fields required by the [[GameEstimator]] for training
   */
  private var rowInputColumnNames: InputColumnsNames = InputColumnsNames()

  /**
   * Column names of the feature shards used by the [[GameEstimator]] for training
   */
  private var featureShardColumnNames: Set[FeatureShardId] = Set()

  /**
   * Training task
   */
  private var taskType: TaskType = TaskType.NONE

  /**
   * Coordinate update ordering
   */
  private var updatingSequence: Seq[CoordinateId] = Seq()

  /**
   * Data configurations for fixed effect coordinates
   */
  private var fixedEffectDataConfigurations: Map[CoordinateId, FixedEffectDataConfiguration] = Map()

  /**
   * Data configurations for random effect coordinates
   */
  private var randomEffectDataConfigurations: Map[CoordinateId, RandomEffectDataConfiguration] = Map()

  /**
   * Number of coordinate descent iterations
   */
  private var numOuterIterations: Int = 1

  /**
   * Compute coefficient variance option
   */
  private var computeVariance: Boolean = false

  /**
   * Optional validation evaluators
   */
  private var evaluatorTypes: Option[Seq[EvaluatorType]] = None

  /**
   * Optional normalization
   */
  private var normalizationContexts: Option[Map[CoordinateId, Broadcast[NormalizationContext]]] = None

  //
  // Setters
  //

  def setDatumInputColumnNames(value: InputColumnsNames): GameEstimator = {
    rowInputColumnNames = value
    this
  }

  def setFeatureShardColumnNames(value: Set[FeatureShardId]): GameEstimator = {
    featureShardColumnNames = value
    this
  }

  def setTaskType(value: TaskType): GameEstimator = {
    taskType = value
    this
  }

  def setUpdatingSequence(value: Seq[CoordinateId]): GameEstimator = {
    updatingSequence = value
    this
  }

  def setFixedEffectDataConfigurations(value: Map[CoordinateId, FixedEffectDataConfiguration]): GameEstimator = {
    fixedEffectDataConfigurations = value
    this
  }

  def setRandomEffectDataConfigurations(value: Map[CoordinateId, RandomEffectDataConfiguration]): GameEstimator = {
    randomEffectDataConfigurations = value
    this
  }

  def setNumOuterIterations(value: Int): GameEstimator = {
    numOuterIterations = value
    this
  }

  def setComputeVariance(value: Boolean): GameEstimator = {
    computeVariance = value
    this
  }

  def setEvaluatorTypes(value: Option[Seq[EvaluatorType]]): GameEstimator = {
    evaluatorTypes = value
    this
  }

  def setNormalizationContexts(value: Option[Map[CoordinateId, Broadcast[NormalizationContext]]]): GameEstimator = {
    normalizationContexts = value
    this
  }

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
      optimizationConfigurations: Seq[GameModelOptimizationConfiguration])
    : Seq[(GameModel, Option[EvaluationResults], GameModelOptimizationConfiguration)] = {

    // Group additional columns to include in GameDatum
    val randomEffectIdCols: Set[String] = randomEffectDataConfigurations.values.map(_.randomEffectType).toSet
    val evaluatorCols = evaluatorTypes.map(ShardedEvaluatorType.getShardedEvaluatorTypeColumns).getOrElse(Set())
    val additionalCols = randomEffectIdCols ++ evaluatorCols

    // Transform the GAME dataset into fixed and random effect specific datasets
    val (trainingDataSets, trainingLossFunctionEvaluator) = prepareTrainingDataSetsAndEvaluator(data, additionalCols)
    val validationDataSetAndEvaluators = prepareValidationDataSetAndEvaluators(validationData, additionalCols)

    val results = optimizationConfigurations.map { modelConfig =>
      Timed(s"Train model with the following config:\n$modelConfig\n") {
        val (gameModel, evaluation) = train(
          modelConfig,
          trainingDataSets,
          trainingLossFunctionEvaluator,
          validationDataSetAndEvaluators)

        (gameModel, evaluation, modelConfig)
      }
    }

    // Purge the training set
    trainingDataSets.foreach { case (_, dataset) =>
      dataset match {
        case rddLike: RDDLike => rddLike.unpersistRDD()
        case _ =>
      }
      dataset match {
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
   * @param additionalCols A set of additional columns whose values should be maintained for training
   * @return A (map of training [[DataSet]]s (one per coordinate), training loss [[Evaluator]]) tuple
   */
  protected def prepareTrainingDataSetsAndEvaluator(
      data: DataFrame,
      additionalCols: Set[String]): (Map[CoordinateId, DataSet[_]], Evaluator) = {

    val numPartitions = data.rdd.getNumPartitions
    val gameDataPartitioner = new LongHashPartitioner(numPartitions)

    val gameDataSet = Timed("Process training data from raw dataframe to RDD of samples") {
      GameConverters
        .getGameDataSetFromDataFrame(
          data,
          featureShardColumnNames,
          additionalCols,
          isResponseRequired = true,
          rowInputColumnNames)
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
  protected def prepareTrainingDataSets(gameDataSet: RDD[(UniqueSampleId, GameDatum)]): Map[CoordinateId, DataSet[_]] = {

    val fixedEffectDataSets = fixedEffectDataConfigurations.map {
      case (id, fixedEffectDataConfiguration) =>
        val fixedEffectDataSet = FixedEffectDataSet.buildWithConfiguration(gameDataSet, fixedEffectDataConfiguration)
          .setName(s"Fixed effect data set with id $id")
          .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
        logger.debug(s"Fixed effect data set with id $id summary:\n${fixedEffectDataSet.toSummaryString}\n")
        (id, fixedEffectDataSet)
    }

    val randomEffectPartitionerMap = randomEffectDataConfigurations.map {
      case (id, randomEffectDataConfiguration) =>
        val numPartitions = randomEffectDataConfiguration.numPartitions
        val randomEffectId = randomEffectDataConfiguration.randomEffectType
        (id, RandomEffectDataSetPartitioner.generateRandomEffectDataSetPartitionerFromGameDataSet(
          numPartitions,
          randomEffectId,
          gameDataSet))
    }

    val randomEffectDataSets = randomEffectDataConfigurations.map {
      case (id, randomEffectDataConfiguration) =>
        val randomEffectPartitioner = randomEffectPartitionerMap(id)
        val rawRandomEffectDataSet =
          RandomEffectDataSet(gameDataSet, randomEffectDataConfiguration, randomEffectPartitioner)
            .setName(s"Random effect data set with coordinate id $id")
            .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
            .materialize()
        val projectorType = randomEffectDataConfiguration.projectorType
        val randomEffectDataSet = projectorType match {
          case IdentityProjection => rawRandomEffectDataSet
          case _ =>
            val randomEffectDataSetInProjectedSpace = RandomEffectDataSetInProjectedSpace
              .buildWithProjectorType(rawRandomEffectDataSet, projectorType)
              .setName(s"Random effect data set in projected space with coordinate id $id")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
              .materialize()
            // Only un-persist the active data and passive data, because randomEffectDataSet and
            // randomEffectDataSetInProjectedSpace share uniqueIdToRandomEffectIds and other RDDs/Broadcasts
            rawRandomEffectDataSet.activeData.unpersist()
            rawRandomEffectDataSet.passiveDataOption.foreach(_.unpersist())
            randomEffectDataSetInProjectedSpace
        }
        logger.debug(s"Random effect data set with id $id summary:\n${randomEffectDataSet.toSummaryString}\n")
        (id, randomEffectDataSet)
    }

    randomEffectPartitionerMap.foreach(_._2.unpersistBroadcast())

    fixedEffectDataSets ++ randomEffectDataSets
  }

  /**
   * Construct the training loss evaluator.
   *
   * @param gameDataSet The training data samples
   * @return A training loss evaluator for the given training task and data
   */
  protected def prepareTrainingLossEvaluator(gameDataSet: RDD[(UniqueSampleId, GameDatum)]): Evaluator = {

    val labelAndOffsetAndWeights = gameDataSet.mapValues(gameData =>
      (gameData.response, gameData.offset, gameData.weight))
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
      additionalCols: Set[String]): Option[(RDD[(UniqueSampleId, GameDatum)], Seq[Evaluator])] =

    dataOpt.map { data =>
      val partitioner = new LongHashPartitioner(data.rdd.partitions.length)
      val gameDataSet = Timed("Convert training data from raw dataframe to processed RDD") {
        val result = GameConverters
          .getGameDataSetFromDataFrame(
            data,
            featureShardColumnNames,
            additionalCols,
            isResponseRequired = true,
            rowInputColumnNames)
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

    evaluatorTypes
      .map(_.map(EvaluatorFactory.buildEvaluator(_, gameDataSet)))
      .getOrElse {
        // Get default evaluators given the task type
        val defaultEvaluator =
          taskType match {
            case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
              new AreaUnderROCCurveEvaluator(validatingLabelsAndOffsetsAndWeights)
            case TaskType.LINEAR_REGRESSION =>
              new RMSEEvaluator(validatingLabelsAndOffsetsAndWeights)
            case TaskType.POISSON_REGRESSION =>
              new PoissonLossEvaluator(validatingLabelsAndOffsetsAndWeights)
            case _ =>
              throw new UnsupportedOperationException(s"$taskType is not a valid GAME training task")
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
      configuration: GameModelOptimizationConfiguration,
      trainingDataSets: Map[CoordinateId, DataSet[_]],
      trainingEvaluator: Evaluator,
      validationDataAndEvaluators: Option[(RDD[(Long, GameDatum)], Seq[Evaluator])])
    : (GameModel, Option[EvaluationResults]) = {

    Timed(s"Train model with the following config:\n$configuration\n") {

      val GameModelOptimizationConfiguration(
          fixedEffectOptimizationConfigurations,
          randomEffectOptimizationConfigurations,
          factoredRandomEffectOptimizationConfigurations) =
        configuration

      val glmConstructor = taskType match {
        case TaskType.LOGISTIC_REGRESSION => LogisticRegressionModel.apply _
        case TaskType.LINEAR_REGRESSION => LinearRegressionModel.apply _
        case TaskType.POISSON_REGRESSION => PoissonRegressionModel.apply _
        case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => SmoothedHingeLossLinearSVMModel.apply _
        case _ => throw new Exception("Need to specify a valid loss function")
      }

      // For each model, create optimization coordinates for the fixed effect, random effect, and factored random effect
      // models
      val coordinates = updatingSequence.map { coordinateId =>
        val coordinate = trainingDataSets(coordinateId) match {

          case fixedEffectDataSet: FixedEffectDataSet =>
            val optimizationConfiguration = fixedEffectOptimizationConfigurations(coordinateId)
            // If number of features is from moderate to large (>200000), then use tree aggregate,
            // otherwise use aggregate.
            val treeAggregateDepth = if (fixedEffectDataSet.numFeatures < FIXED_EFFECT_FEATURE_THRESHOLD) {
              DEFAULT_TREE_AGGREGATE_DEPTH
            } else {
              DEEP_TREE_AGGREGATE_DEPTH
            }

            new FixedEffectCoordinate(
              fixedEffectDataSet,
              DistributedOptimizationProblem(
                optimizationConfiguration,
                selectDistributedLossFunction(optimizationConfiguration, treeAggregateDepth),
                setupDownSampler(optimizationConfiguration.downSamplingRate),
                glmConstructor,
                normalizationContexts.extractOrElse(coordinateId)(defaultNormalizationContext),
                TRACK_STATE,
                computeVariance))

          case randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace =>
            val optimizationConfiguration = randomEffectOptimizationConfigurations(coordinateId)
            new RandomEffectCoordinateInProjectedSpace(
              randomEffectDataSetInProjectedSpace,
              RandomEffectOptimizationProblem(
                  randomEffectDataSetInProjectedSpace,
                  optimizationConfiguration,
                  selectSingleNodeLossFunction(optimizationConfiguration),
                  glmConstructor,
                  normalizationContexts.extractOrElse(coordinateId)(defaultNormalizationContext),
                  TRACK_STATE,
                  computeVariance)
                .setName(s"Random effect optimization problem of coordinate $coordinateId")
                .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL))

          case randomEffectDataSet: RandomEffectDataSet =>
            val FactoredRandomEffectOptimizationConfiguration(randomEffectOptimizationConfiguration,
            latentFactorOptimizationConfiguration,
            mfOptimizationConfiguration) = factoredRandomEffectOptimizationConfigurations(coordinateId)
            val (randomObjectiveFunction, latentObjectiveFunction) =
              selectRandomLatentObjectiveFunction(randomEffectOptimizationConfiguration,
                latentFactorOptimizationConfiguration)
            new FactoredRandomEffectCoordinate(
              randomEffectDataSet,
              FactoredRandomEffectOptimizationProblem(
                  randomEffectDataSet,
                  randomEffectOptimizationConfiguration,
                  latentFactorOptimizationConfiguration,
                  mfOptimizationConfiguration,
                  randomObjectiveFunction,
                  latentObjectiveFunction,
                  setupDownSampler(latentFactorOptimizationConfiguration.downSamplingRate),
                  glmConstructor,
                  normalizationContexts.extractOrElse(coordinateId)(defaultNormalizationContext),
                  TRACK_STATE,
                  computeVariance)
                .setName(s"Factored random effect optimization problem of coordinate $coordinateId")
                .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL))

          case dataSet =>
            throw new UnsupportedOperationException(s"Data set of type ${dataSet.getClass} is not supported")
        }

        Pair[CoordinateId, Coordinate[_]](coordinateId, coordinate)
      }

      CoordinateDescent(coordinates, trainingEvaluator, validationDataAndEvaluators, logger).run(numOuterIterations)
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
      taskType match {
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
   * @return A (
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
      DEFAULT_TREE_AGGREGATE_DEPTH)

    taskType match {
      case TaskType.LOGISTIC_REGRESSION => (lossFunction(LogisticLossFunction), distLossFunction(LogisticLossFunction))
      case TaskType.LINEAR_REGRESSION => (lossFunction(SquaredLossFunction), distLossFunction(SquaredLossFunction))
      case TaskType.POISSON_REGRESSION => (lossFunction(PoissonLossFunction), distLossFunction(PoissonLossFunction))
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        (SingleNodeSmoothedHingeLossFunction(randomEffectOptimizationConfiguration),
          DistributedSmoothedHingeLossFunction(sc, latentFactorOptimizationConfiguration, DEFAULT_TREE_AGGREGATE_DEPTH))
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

    taskType match {
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

    taskType match {
      case TaskType.LOGISTIC_REGRESSION => distLossFunction(LogisticLossFunction)
      case TaskType.LINEAR_REGRESSION => distLossFunction(SquaredLossFunction)
      case TaskType.POISSON_REGRESSION => distLossFunction(PoissonLossFunction)
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        DistributedSmoothedHingeLossFunction(sc, configuration, treeAggregateDepth)
    }
  }
}

object GameEstimator {
  val FIXED_EFFECT_FEATURE_THRESHOLD = 200000
  val DEFAULT_TREE_AGGREGATE_DEPTH = 1
  val DEEP_TREE_AGGREGATE_DEPTH = 2
  val TRACK_STATE = true
}
