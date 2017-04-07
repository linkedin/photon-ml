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
import com.linkedin.photon.ml.Types.FeatureShardId
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
 * @param params Configuration parameters for the estimator
 * @param sc The spark context
 * @param logger The logger instance
 */
class GameEstimator(val sc: SparkContext, val params: GameParams, implicit val logger: Logger) {

  import GameEstimator._

  // 2 types that makes the code more readable
  type LossFunction = (PointwiseLossFunction) => SingleNodeGLMLossFunction
  type DistributedLossFunction = (PointwiseLossFunction) => DistributedGLMLossFunction

  protected[estimators] val idTypeSet: Set[String] =
    params.randomEffectDataConfigurations.values.map(_.randomEffectType).toSet ++ params.getShardedEvaluatorIdTypes

  protected[estimators] val defaultNormalizationContext: Broadcast[NormalizationContext] =
    sc.broadcast(NoNormalization())

  /**
   * Fits GAME models to the training dataset.
   *
   * @param data The training set
   * @param normalizationContexts Optional training data statistics used e.g. for normalization, for each feature shard
   * @param validationData Optional validation set for per-iteration validation
   * @return A set of GAME models, one for each combination of fixed and random effect combination specified in the
   *         params
   */
  def fit(
      data: DataFrame,
      validationData: Option[DataFrame] = None,
      normalizationContexts: Option[Map[FeatureShardId, NormalizationContext]])
    : Seq[(GameModel, Option[EvaluationResults], GameModelOptimizationConfiguration)] = {

    val numPartitions = data.rdd.partitions.length
    val gameDataPartitioner = new LongHashPartitioner(numPartitions)

    val gameDataSet = Timed("Convert data set from dataframe") {
      GameConverters
        .getGameDataSetFromDataFrame(data,
          params.featureShardIdToFeatureSectionKeysMap.keys.toSet,
          idTypeSet,
          isResponseRequired = true,
          params.inputColumnsNames)
        .partitionBy(gameDataPartitioner)
        .setName("GAME training data")
        .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    }
    gameDataSet.count()

    // Transform the GAME dataset into fixed and random effect specific datasets
    val trainingDataSet = Timed("Prepare training data") { prepareTrainingDataSet(gameDataSet) }

    val trainingLossFunctionEvaluator =
      Timed("Prepare training loss evaluator") { prepareTrainingLossEvaluator(gameDataSet) }

    // Purge the GAME dataset, which is no longer needed in the following code
    gameDataSet.unpersist()

    val validationDataAndEvaluators = Timed("Prepare validation evaluators") {
      validationData.map(prepareValidationEvaluators)
    }

    val gameModelsMap = Timed("Train model") {
      train(trainingDataSet, trainingLossFunctionEvaluator, validationDataAndEvaluators, normalizationContexts)
    }

    // Purge the training set
    trainingDataSet.foreach { case (_, dataset) =>
      dataset match {
        case rddLike: RDDLike => rddLike.unpersistRDD()
        case _ =>
      }
      dataset match {
        case broadcastLike: BroadcastLike => broadcastLike.unpersistBroadcast()
        case _ =>
      }
    }

    gameModelsMap
  }

  /**
   * Builds 1 or 2 data sets (depending on parameters) to train the model. These data sets are for:
   * - the fixed effect part of the model,
   * - the random effect parts of the model,
   *
   * @param gameDataSet The input dataset
   * @return The training dataset
   */
  protected[estimators] def prepareTrainingDataSet(
      gameDataSet: RDD[(Long, GameDatum)]): Map[String, DataSet[_ <: DataSet[_]]] = {

    val fixedEffectDataSets = params.fixedEffectDataConfigurations.map {
      case (id, fixedEffectDataConfiguration) =>
        val fixedEffectDataSet = FixedEffectDataSet.buildWithConfiguration(gameDataSet, fixedEffectDataConfiguration)
          .setName(s"Fixed effect data set with id $id")
          .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
        logger.debug(s"Fixed effect data set with id $id summary:\n${fixedEffectDataSet.toSummaryString}\n")
        (id, fixedEffectDataSet)
    }

    val randomEffectPartitionerMap = params.randomEffectDataConfigurations.map {
      case (id, randomEffectDataConfiguration) =>
        val numPartitions = randomEffectDataConfiguration.numPartitions
        val randomEffectId = randomEffectDataConfiguration.randomEffectType
        (id, RandomEffectDataSetPartitioner.generateRandomEffectDataSetPartitionerFromGameDataSet(
          numPartitions,
          randomEffectId,
          gameDataSet))
    }

    val randomEffectDataSets = params.randomEffectDataConfigurations.map {
      case (id, randomEffectDataConfiguration) =>
        val randomEffectPartitioner = randomEffectPartitionerMap(id)
        val rawRandomEffectDataSet = RandomEffectDataSet
          .buildWithConfiguration(gameDataSet, randomEffectDataConfiguration, randomEffectPartitioner)
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
   * Creates the training evaluator.
   *
   * @param gameDataSet The input dataset
   * @return The training evaluator
   */
  protected[estimators] def prepareTrainingLossEvaluator(gameDataSet: RDD[(Long, GameDatum)]): Evaluator = {

    val labelAndOffsetAndWeights = gameDataSet.mapValues(gameData =>
      (gameData.response, gameData.offset, gameData.weight))
      .setName("Training labels, offsets and weights")
      .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)

    labelAndOffsetAndWeights.count()

    params.taskType match {
      case TaskType.LOGISTIC_REGRESSION =>
        new LogisticLossEvaluator(labelAndOffsetAndWeights)
      case TaskType.LINEAR_REGRESSION =>
        new SquaredLossEvaluator(labelAndOffsetAndWeights)
      case TaskType.POISSON_REGRESSION =>
        new PoissonLossEvaluator(labelAndOffsetAndWeights)
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        new SmoothedHingeLossEvaluator(labelAndOffsetAndWeights)
      case _ =>
        throw new UnsupportedOperationException(s"${params.taskType} is not a valid training evaluator")
    }
  }

  /**
   * Creates the validation evaluator(s).
   *
   * @param data The validation data
   * @return The validation data and the companion evaluator
   */
  protected[estimators] def prepareValidationEvaluators(data: DataFrame): (RDD[(Long, GameDatum)], Seq[Evaluator]) = {

    val partitioner = new LongHashPartitioner(data.rdd.partitions.length)
    val gameDataSet =
      GameConverters
        .getGameDataSetFromDataFrame(data,
          params.featureShardIdToFeatureSectionKeysMap.keys.toSet,
          idTypeSet,
          isResponseRequired = true,
          params.inputColumnsNames)
      .partitionBy(partitioner)
      .setName("Validating Game data set")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val validatingLabelsAndOffsetsAndWeights = gameDataSet
      .mapValues(gameData => (gameData.response, gameData.offset, gameData.weight))
      .setName(s"Validating labels and offsets")
      .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    validatingLabelsAndOffsetsAndWeights.count()

    val evaluators =
      if (params.evaluatorTypes.isEmpty) {
        // Get default evaluators given the task type
        val defaultEvaluator =
          params.taskType match {
            case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
              new AreaUnderROCCurveEvaluator(validatingLabelsAndOffsetsAndWeights)
            case TaskType.LINEAR_REGRESSION =>
              new RMSEEvaluator(validatingLabelsAndOffsetsAndWeights)
            case TaskType.POISSON_REGRESSION =>
              new PoissonLossEvaluator(validatingLabelsAndOffsetsAndWeights)
            case _ =>
              throw new UnsupportedOperationException(s"${params.taskType} is not a valid validating evaluator")
          }
        Seq(defaultEvaluator)
      } else {
        params.evaluatorTypes.map(EvaluatorFactory.buildEvaluator(_, gameDataSet))
      }

    val randomScores = gameDataSet.mapValues(_ => math.random)
    evaluators.foreach { evaluator =>
      val metric = evaluator.evaluate(randomScores)
      logger.info(s"Random guessing based baseline evaluation metric for ${evaluator.getEvaluatorName}: $metric")
    }

    (gameDataSet, evaluators)
  }

  /**
   * Train GAME models. This method builds a coordinate descent optimization problem from the individual optimization
   * problems for the fixed effect, random effect, and factored random effect models.
   *
   * @param dataSets The training datasets
   * @param trainingEvaluator The training evaluator
   * @param validationDataAndEvaluators Optional validation dataset and evaluators
   * @param normalizationContexts Optional normalization contexts
   * @return Trained GAME models
   */
  protected[estimators] def train(
      dataSets: Map[String, DataSet[_ <: DataSet[_]]],
      trainingEvaluator: Evaluator,
      validationDataAndEvaluators: Option[(RDD[(Long, GameDatum)], Seq[Evaluator])],
      normalizationContexts: Option[Map[FeatureShardId, NormalizationContext]])
    :Seq[(GameModel, Option[EvaluationResults], GameModelOptimizationConfiguration)] = {

    val contextBroadcasts: Option[Map[FeatureShardId, Broadcast[NormalizationContext]]] = normalizationContexts.map {
      contextsMap => contextsMap.mapValues { context => sc.broadcast(context) }
    }

    val gameModels = for (
        fixedEffectOptimizationConfiguration <- params.fixedEffectOptimizationConfigurations;
        randomEffectOptimizationConfiguration <- params.randomEffectOptimizationConfigurations;
        factoredRandomEffectOptimizationConfiguration <- params.factoredRandomEffectOptimizationConfigurations) yield {

      val modelConfig = GameModelOptimizationConfiguration(
        fixedEffectOptimizationConfiguration,
        randomEffectOptimizationConfiguration,
        factoredRandomEffectOptimizationConfiguration)

      val timer = Timer.start()
      logger.info(s"Start to train the Game model with the following config:\n$modelConfig\n")

      val glmConstructor = params.taskType match {
        case TaskType.LOGISTIC_REGRESSION => LogisticRegressionModel.apply _
        case TaskType.LINEAR_REGRESSION => LinearRegressionModel.apply _
        case TaskType.POISSON_REGRESSION => PoissonRegressionModel.apply _
        case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => SmoothedHingeLossLinearSVMModel.apply _
        case _ => throw new Exception("Need to specify a valid loss function")
      }

      // For each model, create optimization coordinates for the fixed effect, random effect, and factored random effect
      // models
      val coordinates = params.updatingSequence.map { coordinateId =>
        val coordinate = dataSets(coordinateId) match {

          case fixedEffectDataSet: FixedEffectDataSet =>
            val optimizationConfiguration = fixedEffectOptimizationConfiguration(coordinateId)
            val featureShardId = params.fixedEffectDataConfigurations(coordinateId).featureShardId
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
                contextBroadcasts.extractOrElse(featureShardId)(defaultNormalizationContext),
                TRACK_STATE,
                params.computeVariance
              ))

          case randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace =>
            val featureShardId = params.randomEffectDataConfigurations(coordinateId).featureShardId
            val optimizationConfiguration = randomEffectOptimizationConfiguration(coordinateId)
            new RandomEffectCoordinateInProjectedSpace(
              randomEffectDataSetInProjectedSpace,
              RandomEffectOptimizationProblem(
                randomEffectDataSetInProjectedSpace,
                optimizationConfiguration,
                selectSingleNodeLossFunction(optimizationConfiguration),
                glmConstructor,
                contextBroadcasts.extractOrElse(featureShardId)(defaultNormalizationContext),
                TRACK_STATE,
                params.computeVariance)
                .setName(s"Random effect optimization problem of coordinate $coordinateId")
                .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL))

          case randomEffectDataSet: RandomEffectDataSet =>
            val featureShardId = params.randomEffectDataConfigurations(coordinateId).featureShardId
            val FactoredRandomEffectOptimizationConfiguration(randomEffectOptimizationConfiguration,
              latentFactorOptimizationConfiguration,
              mfOptimizationConfiguration) = factoredRandomEffectOptimizationConfiguration(coordinateId)
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
                contextBroadcasts.extractOrElse(featureShardId)(defaultNormalizationContext),
                TRACK_STATE,
                params.computeVariance)
                .setName(s"Factored random effect optimization problem of coordinate $coordinateId")
                .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL))

          case dataSet =>
            throw new UnsupportedOperationException(s"Data set of type ${dataSet.getClass} is not supported")
        }
        Pair[String, Coordinate[_ <: DataSet[_]]](coordinateId, coordinate)
      }

      val (gameModel, evaluation) =
        CoordinateDescent(coordinates, trainingEvaluator, validationDataAndEvaluators, logger)
        .run(params.numIterations)

      timer.stop()
      logger.info(s"Finished training model with the following config:\n$modelConfig\n" +
          s"Time elapsed: ${timer.durationSeconds} (s)\n")

      (gameModel, evaluation, modelConfig)
    }

    gameModels
  }

  /**
   * Sets the sampling option for this model training run.
   *
   * @param downSamplingRate the downSamplingRate
   * @return An Option containing a DownSampler or None
   */
  protected[estimators] def setupDownSampler(downSamplingRate: Double): Option[DownSampler] =

    if (downSamplingRate > 0D && downSamplingRate < 1D) {
      params.taskType match {
        case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
          Some(new BinaryClassificationDownSampler(downSamplingRate))
        case TaskType.LINEAR_REGRESSION | TaskType.POISSON_REGRESSION =>
          Some(new DefaultDownSampler(downSamplingRate))
      }
    } else {
      None
    }

  private def selectRandomLatentObjectiveFunction(
      randomEffectOptimizationConfiguration: GLMOptimizationConfiguration,
      latentFactorOptimizationConfiguration: GLMOptimizationConfiguration):
      (SingleNodeObjectiveFunction, DistributedObjectiveFunction) = {

    val lossFunction: LossFunction  = SingleNodeGLMLossFunction(randomEffectOptimizationConfiguration)
    val distLossFunction: DistributedLossFunction =
      DistributedGLMLossFunction(sc, latentFactorOptimizationConfiguration, DEFAULT_TREE_AGGREGATE_DEPTH)

    params.taskType match {
      case TaskType.LOGISTIC_REGRESSION => (lossFunction(LogisticLossFunction), distLossFunction(LogisticLossFunction))
      case TaskType.LINEAR_REGRESSION => (lossFunction(SquaredLossFunction), distLossFunction(SquaredLossFunction))
      case TaskType.POISSON_REGRESSION => (lossFunction(PoissonLossFunction), distLossFunction(PoissonLossFunction))
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        (SingleNodeSmoothedHingeLossFunction(randomEffectOptimizationConfiguration),
          DistributedSmoothedHingeLossFunction(sc, latentFactorOptimizationConfiguration, DEFAULT_TREE_AGGREGATE_DEPTH))
    }
  }

  private def selectSingleNodeLossFunction(configuration: GLMOptimizationConfiguration): SingleNodeObjectiveFunction = {

    val lossFunction: LossFunction = SingleNodeGLMLossFunction(configuration)

    params.taskType match {
      case TaskType.LOGISTIC_REGRESSION => lossFunction(LogisticLossFunction)
      case TaskType.LINEAR_REGRESSION => lossFunction(SquaredLossFunction)
      case TaskType.POISSON_REGRESSION => lossFunction(PoissonLossFunction)
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => SingleNodeSmoothedHingeLossFunction(configuration)
    }
  }

  private def selectDistributedLossFunction(
      configuration: GLMOptimizationConfiguration,
      treeAggregateDepth: Int): DistributedObjectiveFunction = {

    val distLossFunction: DistributedLossFunction = DistributedGLMLossFunction(sc, configuration, treeAggregateDepth)

    params.taskType match {
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
