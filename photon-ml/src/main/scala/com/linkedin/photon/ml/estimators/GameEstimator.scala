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
package com.linkedin.photon.ml.estimators

import scala.collection.Map

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.slf4j.Logger

import com.linkedin.photon.ml.algorithm._
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.evaluation.Evaluator.EvaluationResults
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.function.glm._
import com.linkedin.photon.ml.function.svm.{DistributedSmoothedHingeLossFunction, SingleNodeSmoothedHingeLossFunction}
import com.linkedin.photon.ml.model.GAMEModel
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.optimization.DistributedOptimizationProblem
import com.linkedin.photon.ml.optimization.game.{FactoredRandomEffectOptimizationProblem, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.projector.IdentityProjection
import com.linkedin.photon.ml.sampler.{BinaryClassificationDownSampler, DefaultDownSampler}
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.util._
import com.linkedin.photon.ml.{BroadcastLike, RDDLike, TaskType}

/**
 * Estimator implementation for GAME models
 *
 * @param params configuration parameters for the estimator
 * @param sparkContext the spark context
 * @param logger the logger instance
 */
class GameEstimator(val params: GameParams, val sparkContext: SparkContext, val logger: Logger) {
  import GameEstimator._

  protected[estimators] val idTypeSet: Set[String] =
    params.randomEffectDataConfigurations.values.map(_.randomEffectType).toSet ++ params.getShardedEvaluatorIdTypes

  protected[estimators] val defaultNormalizationContext: Broadcast[NormalizationContext] =
    sparkContext.broadcast(NoNormalization())

  /**
   * Fits GAME models to the training dataset
   *
   * @param data the training set
   * @param validatingData optional validation set for per-iteration validation
   * @return a set of GAME models, one for each combination of fixed and random effect combination specified in the
   *   params
   */
  def fit(
      data: DataFrame,
      validatingData: Option[DataFrame] = None): Seq[(GAMEModel, Option[EvaluationResults], String)] = {

    val timer = new Timer

    // Convert the dataframe to a GAME dataset
    val numPartitions = data.rdd.partitions.length
    val gameDataPartitioner = new LongHashPartitioner(numPartitions)
    val gameDataSet = GameConverters.getGameDataSetFromDataFrame(
      data,
      params.featureShardIdToFeatureSectionKeysMap.keys.toSet,
      idTypeSet,
      isResponseRequired = true)
      .partitionBy(gameDataPartitioner)
      .setName("GAME training data")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    gameDataSet.count()

    logger.info(s"Time elapsed after converting data to GAME dataset: ${timer.durationSeconds} (s)\n")

    // Transform the GAME dataset into fixed and random effect specific datasets
    timer.start()
    val trainingDataSet = prepareTrainingDataSet(gameDataSet)
    timer.stop()

    logger.info(s"Time elapsed after training data set preparation: ${timer.durationSeconds} (s)\n")

    // Prepare training loss evaluator
    timer.start()
    val trainingLossFunctionEvaluator = prepareTrainingLossEvaluator(gameDataSet)
    timer.stop()
    logger.info(s"Time elapsed after training evaluator preparation: ${timer.durationSeconds} (s)\n")

    // Purge the GAME dataset, which is no longer needed in the following code
    gameDataSet.unpersist()

    // Prepare the validating evaluator
    val validatingDataAndEvaluators = validatingData.map { data =>
      timer.start()
      val validatingDataAndEvaluators = prepareValidatingEvaluators(data)
      timer.stop()

      logger.info("Time elapsed after validating data and evaluator preparation: " +
        s"${timer.durationSeconds} (s)\n")

      validatingDataAndEvaluators
    }

    // Train models
    timer.start()
    val gameModelsMap = train(trainingDataSet, trainingLossFunctionEvaluator, validatingDataAndEvaluators)
    timer.stop()

    logger.info(s"Time elapsed after game model training: ${timer.durationSeconds} (s)\n")

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
   * Builds the training fixed effect and random effect specific datasets from the input GAME dataset
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

    // Prepare the per-random effect partitioner
    val randomEffectPartitionerMap = params.randomEffectDataConfigurations.map {
      case (id, randomEffectDataConfiguration) =>
        val numPartitions = randomEffectDataConfiguration.numPartitions
        val randomEffectId = randomEffectDataConfiguration.randomEffectType
        (id, RandomEffectDataSetPartitioner.generateRandomEffectDataSetPartitionerFromGameDataSet(
          numPartitions,
          randomEffectId,
          gameDataSet))
    }

    // Prepare the random effect data sets
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
   * Creates the training evaluator
   *
   * @param gameDataSet The input dataset
   * @return The training evaluator
   */
  protected[estimators] def prepareTrainingLossEvaluator(gameDataSet: RDD[(Long, GameDatum)]): Evaluator = {

    val labelAndOffsetAndWeights = gameDataSet.mapValues(gameData =>
      (gameData.response, gameData.offset, gameData.weight))
      .setName("Training label and offset and weights")
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
   * Creates the validation evaluator(s)
   *
   * @param data The input data
   * @return The validating game data sets and the companion evaluator
   */
  protected[estimators] def prepareValidatingEvaluators(
      data: DataFrame): (RDD[(Long, GameDatum)], Seq[Evaluator]) = {

    val partitioner = new LongHashPartitioner(data.rdd.partitions.length)
    val gameDataSet = GameConverters.getGameDataSetFromDataFrame(
      data,
      params.featureShardIdToFeatureSectionKeysMap.keys.toSet,
      idTypeSet,
      isResponseRequired = true)
      .partitionBy(partitioner)
      .setName("Validating Game data set")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val validatingLabelsAndOffsetsAndWeights = gameDataSet
      .mapValues(gameData => (gameData.response, gameData.offset, gameData.weight))
      .setName(s"Validating labels and offsets").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
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
        params.evaluatorTypes.map(Evaluator.buildEvaluator(_, gameDataSet))
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
   * @param validatingDataAndEvaluatorsOption Optional validation dataset and evaluators
   * @return trained GAME models
   */
  protected[estimators] def train(
      dataSets: Map[String, DataSet[_ <: DataSet[_]]],
      trainingEvaluator: Evaluator,
      validatingDataAndEvaluatorsOption: Option[(RDD[(Long, GameDatum)], Seq[Evaluator])]):
        Seq[(GAMEModel, Option[EvaluationResults], String)] = {

    val gameModels = for (
        fixedEffectOptimizationConfiguration <- params.fixedEffectOptimizationConfigurations;
        randomEffectOptimizationConfiguration <- params.randomEffectOptimizationConfigurations;
        factoredRandomEffectOptimizationConfiguration <- params.factoredRandomEffectOptimizationConfigurations) yield {

      // TODO: this is only geared toward readability by humans, but we need to save those in machine readable format
      // as model metadata too
      val modelConfig = fixedEffectOptimizationConfiguration.mkString("\n") + "\n" +
          randomEffectOptimizationConfiguration.mkString("\n") + "\n" +
          factoredRandomEffectOptimizationConfiguration.mkString("\n")

      val timer = Timer.start()
      logger.info(s"Start to train the game model with the following config:\n$modelConfig\n")

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
          // Fixed effect coordinate
          case fixedEffectDataSet: FixedEffectDataSet =>
            val optimizationConfiguration = fixedEffectOptimizationConfiguration(coordinateId)
            val downSamplingRate = optimizationConfiguration.downSamplingRate
            // If number of features is from moderate to large (>200000), then use tree aggregate,
            // otherwise use aggregate.
            val treeAggregateDepth = if (fixedEffectDataSet.numFeatures < FIXED_EFFECT_FEATURE_THRESHOLD) {
              DEFAULT_TREE_AGGREGATE_DEPTH
            } else {
              DEEP_TREE_AGGREGATE_DEPTH
            }
            val objectiveFunction = params.taskType match {
              case TaskType.LOGISTIC_REGRESSION =>
                DistributedGLMLossFunction.create(
                  optimizationConfiguration,
                  LogisticLossFunction,
                  sparkContext,
                  treeAggregateDepth)

              case TaskType.LINEAR_REGRESSION =>
                DistributedGLMLossFunction.create(
                  optimizationConfiguration,
                  SquaredLossFunction,
                  sparkContext,
                  treeAggregateDepth)

              case TaskType.POISSON_REGRESSION =>
                DistributedGLMLossFunction.create(
                  optimizationConfiguration,
                  PoissonLossFunction,
                  sparkContext,
                  treeAggregateDepth)

              case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
                DistributedSmoothedHingeLossFunction.create(
                  optimizationConfiguration,
                  sparkContext,
                  treeAggregateDepth)
            }
            val samplerOption = if (downSamplingRate > 0D && downSamplingRate < 1D) {
              params.taskType match {
                case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
                  Some(new BinaryClassificationDownSampler(downSamplingRate))
                case TaskType.LINEAR_REGRESSION | TaskType.POISSON_REGRESSION =>
                  Some(new DefaultDownSampler(downSamplingRate))
              }
            } else {
              None
            }
            val optimizationProblem = DistributedOptimizationProblem.create(
              optimizationConfiguration,
              objectiveFunction,
              samplerOption,
              glmConstructor,
              defaultNormalizationContext,
              TRACK_STATE,
              params.computeVariance
            )

            new FixedEffectCoordinate(fixedEffectDataSet, optimizationProblem)

          case randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace =>
            // Random effect coordinate
            val optimizationConfiguration = randomEffectOptimizationConfiguration(coordinateId)
            val objectiveFunction = params.taskType match {
              case TaskType.LOGISTIC_REGRESSION =>
                SingleNodeGLMLossFunction.create(
                  optimizationConfiguration,
                  LogisticLossFunction)

              case TaskType.LINEAR_REGRESSION =>
                SingleNodeGLMLossFunction.create(
                  optimizationConfiguration,
                  SquaredLossFunction)

              case TaskType.POISSON_REGRESSION =>
                SingleNodeGLMLossFunction.create(
                  optimizationConfiguration,
                  PoissonLossFunction)

              case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
                SingleNodeSmoothedHingeLossFunction.create(
                  optimizationConfiguration)
            }
            val randomEffectOptimizationProblem = RandomEffectOptimizationProblem
              .create(
                randomEffectDataSetInProjectedSpace,
                optimizationConfiguration,
                objectiveFunction,
                glmConstructor,
                defaultNormalizationContext,
                TRACK_STATE,
                params.computeVariance)
              .setName(s"Random effect optimization problem of coordinate $coordinateId")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

            new RandomEffectCoordinateInProjectedSpace(
              randomEffectDataSetInProjectedSpace,
              randomEffectOptimizationProblem)

          case randomEffectDataSet: RandomEffectDataSet =>
            // Factored random effect coordinate
            val (randomEffectOptimizationConfiguration,
              latentFactorOptimizationConfiguration,
              mfOptimizationConfiguration) = factoredRandomEffectOptimizationConfiguration(coordinateId)

            val downSamplingRate = latentFactorOptimizationConfiguration.downSamplingRate
            val samplerOption = if (downSamplingRate > 0D && downSamplingRate < 1D) {
              params.taskType match {
                case TaskType.LOGISTIC_REGRESSION | TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
                  Some(new BinaryClassificationDownSampler(downSamplingRate))
                case TaskType.LINEAR_REGRESSION | TaskType.POISSON_REGRESSION =>
                  Some(new DefaultDownSampler(downSamplingRate))
              }
            } else {
              None
            }
            val (randomObjectiveFunction, latentObjectiveFunction) = params.taskType match {
              case TaskType.LOGISTIC_REGRESSION =>
                val random = SingleNodeGLMLossFunction.create(
                  randomEffectOptimizationConfiguration,
                  LogisticLossFunction)
                val latent = DistributedGLMLossFunction.create(
                  latentFactorOptimizationConfiguration,
                  LogisticLossFunction,
                  sparkContext,
                  DEFAULT_TREE_AGGREGATE_DEPTH)
                (random, latent)

              case TaskType.LINEAR_REGRESSION =>
                val random = SingleNodeGLMLossFunction.create(
                  randomEffectOptimizationConfiguration,
                  SquaredLossFunction)
                val latent = DistributedGLMLossFunction.create(
                  latentFactorOptimizationConfiguration,
                  SquaredLossFunction,
                  sparkContext,
                  DEFAULT_TREE_AGGREGATE_DEPTH)
                (random, latent)

              case TaskType.POISSON_REGRESSION =>
                val random = SingleNodeGLMLossFunction.create(
                  randomEffectOptimizationConfiguration,
                  PoissonLossFunction)
                val latent = DistributedGLMLossFunction.create(
                  latentFactorOptimizationConfiguration,
                  PoissonLossFunction,
                  sparkContext,
                  DEFAULT_TREE_AGGREGATE_DEPTH)
                (random, latent)

              case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
                val random = SingleNodeSmoothedHingeLossFunction.create(
                  randomEffectOptimizationConfiguration)
                val latent = DistributedSmoothedHingeLossFunction.create(
                  latentFactorOptimizationConfiguration,
                  sparkContext,
                  DEFAULT_TREE_AGGREGATE_DEPTH)
                (random, latent)
            }
            val factoredRandomEffectOptimizationProblem = FactoredRandomEffectOptimizationProblem
              .create(
                randomEffectDataSet,
                randomEffectOptimizationConfiguration,
                latentFactorOptimizationConfiguration,
                mfOptimizationConfiguration,
                randomObjectiveFunction,
                latentObjectiveFunction,
                samplerOption,
                glmConstructor,
                defaultNormalizationContext,
                TRACK_STATE,
                params.computeVariance)
              .setName(s"Factored random effect optimization problem of coordinate $coordinateId")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

            new FactoredRandomEffectCoordinate(randomEffectDataSet, factoredRandomEffectOptimizationProblem)

          case dataSet =>
            throw new UnsupportedOperationException(s"Data set of type ${dataSet.getClass} is not supported")
        }
        Pair[String, Coordinate[_ <: DataSet[_], _ <: Coordinate[_, _]]](coordinateId, coordinate)
      }
      val coordinateDescent =
        new CoordinateDescent(coordinates, trainingEvaluator, validatingDataAndEvaluatorsOption, logger)
      val (gameModel, evaluation) = coordinateDescent.run(params.numIterations, params.taskType)

      timer.stop()
      logger.info(s"Finished training model with the following config:\n$modelConfig\n" +
          s"Time elapsed: ${timer.durationSeconds} (s)\n")

      (gameModel, evaluation, modelConfig)
    }

    gameModels
  }
}

object GameEstimator {
  val FIXED_EFFECT_FEATURE_THRESHOLD = 200000
  val DEFAULT_TREE_AGGREGATE_DEPTH = 1
  val DEEP_TREE_AGGREGATE_DEPTH = 2
  val TRACK_STATE = true
}
