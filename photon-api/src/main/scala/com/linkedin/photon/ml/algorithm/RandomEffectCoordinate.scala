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
package com.linkedin.photon.ml.algorithm

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.{REId, UniqueSampleId}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.{DatumScoringModel, RandomEffectModel}
import com.linkedin.photon.ml.optimization.game.RandomEffectOptimizationProblem
import com.linkedin.photon.ml.optimization.{OptimizationTracker, RandomEffectOptimizationTracker}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * The optimization problem coordinate for a random effect model.
 *
 * @tparam Objective The type of objective function used to solve individual random effect optimization problems
 * @param dataset The training dataset
 * @param optimizationProblem The random effect optimization problem
 */
protected[ml] abstract class RandomEffectCoordinate[Objective <: SingleNodeObjectiveFunction](
    dataset: RandomEffectDataset,
    optimizationProblem: RandomEffectOptimizationProblem[Objective])
  extends Coordinate[RandomEffectDataset](dataset) {

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A tuple of the updated model and the optimization states tracker
   */
  override protected[algorithm] def trainModel(): (DatumScoringModel, Option[OptimizationTracker]) =
    RandomEffectCoordinate.trainModel(dataset, optimizationProblem, None)

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A tuple of the updated model and the optimization states tracker
   */
  override protected[algorithm] def trainModel(
      model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker]) =
    model match {
      case randomEffectModel: RandomEffectModel =>
        RandomEffectCoordinate.trainModel(dataset, optimizationProblem, Some(randomEffectModel))

      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} " +
            s"in ${this.getClass} is not supported")
    }

  /**
   * Score the effect-specific dataset in the coordinate with the input model.
   *
   * @param model The input model
   * @return The output scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores =
    model match {
      case randomEffectModel: RandomEffectModel => RandomEffectCoordinate.score(dataset, randomEffectModel)

      case _ => throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
        s"in ${this.getClass} is not supported")
    }
}

object RandomEffectCoordinate {

  /**
   * Train a new [[RandomEffectModel]] (i.e. run model optimization for each entity).
   *
   * @tparam Function The type of objective function used to solve individual random effect optimization problems
   * @param randomEffectDataset The training dataset
   * @param randomEffectOptimizationProblem The per-entity optimization problems
   * @param initialRandomEffectModelOpt An optional existing [[RandomEffectModel]] to use as a starting point for
   *                                    optimization
   * @return A (new [[RandomEffectModel]], optional optimization stats) tuple
   */
  protected[algorithm] def trainModel[Function <: SingleNodeObjectiveFunction](
      randomEffectDataset: RandomEffectDataset,
      randomEffectOptimizationProblem: RandomEffectOptimizationProblem[Function],
      initialRandomEffectModelOpt: Option[RandomEffectModel])
    : (RandomEffectModel, Option[RandomEffectOptimizationTracker]) = {

    val dataAndOptimizationProblems = randomEffectDataset
      .activeData
      .join(randomEffectOptimizationProblem.optimizationProblems)

    // Left join the models to data and optimization problems for cases where we have a prior model but no new data
    val newModelsAndTrackers = initialRandomEffectModelOpt
      .map { randomEffectModel =>
        randomEffectModel
          .modelsRDD
          .leftOuterJoin(dataAndOptimizationProblems)
          .mapValues {
            case (localModel, Some((localDataset, optimizationProblem))) =>
              val trainingLabeledPoints = localDataset.dataPoints.map(_._2)
              val updatedModel = optimizationProblem.run(trainingLabeledPoints, localModel)
              val stateTrackers = optimizationProblem.getStatesTracker

              (updatedModel, stateTrackers)

            case (localModel, _) =>
              (localModel, None)
          }
      }
      .getOrElse {
        dataAndOptimizationProblems.mapValues { case (localDataset, optimizationProblem) =>
          val trainingLabeledPoints = localDataset.dataPoints.map(_._2)
          val newModel = optimizationProblem.run(trainingLabeledPoints)
          val stateTrackers = optimizationProblem.getStatesTracker

          (newModel, stateTrackers)
        }
      }
      .setName(s"Updated models and state trackers for random effect ${randomEffectDataset.randomEffectType}")
      .persist(StorageLevel.MEMORY_ONLY)

    val newRandomEffectModel = new RandomEffectModel(
      newModelsAndTrackers.mapValues(_._1),
      randomEffectDataset.randomEffectType,
      randomEffectDataset.featureShardId)

    val optimizationTracker: Option[RandomEffectOptimizationTracker] =
      if (randomEffectOptimizationProblem.isTrackingState) {
        val stateTrackers = newModelsAndTrackers.flatMap(_._2._2)
        val randomEffectTracker = RandomEffectOptimizationTracker(stateTrackers)

        Some(randomEffectTracker)

      } else {
        None
      }

    (newRandomEffectModel, optimizationTracker)
  }

  /**
   * Score a [[RandomEffectDataset]] using a given [[RandomEffectModel]].
   *
   * For information about the differences between active and passive data, see the [[RandomEffectDataset]]
   * documentation.
   *
   * @note The score is the dot product of the model coefficients with the feature values (i.e., it does not go
   *       through a non-linear link function).
   * @param randomEffectDataset The dataset to score
   * @param randomEffectModel The model used to score the dataset
   * @return The computed scores
   */
  protected[algorithm] def score(
    randomEffectDataset: RandomEffectDataset,
    randomEffectModel: RandomEffectModel): CoordinateDataScores = {

    val activeScores = randomEffectDataset
      .activeData
      .join(randomEffectModel.modelsRDD)
      .flatMap { case (_, (localDataset, model)) =>
        localDataset.dataPoints.map { case (uniqueId, labeledPoint) =>
          (uniqueId, model.computeScore(labeledPoint.features))
        }
      }
      .partitionBy(randomEffectDataset.uniqueIdPartitioner)

    val passiveScores = computePassiveScores(
        randomEffectDataset.passiveData,
        randomEffectDataset.passiveDataRandomEffectIds,
        randomEffectModel.modelsRDD)

    new CoordinateDataScores(activeScores ++ passiveScores)
  }

  /**
   * Computes passive scores.
   *
   * For information about the differences between active and passive data, see the [[RandomEffectDataset]]
   * documentation.
   *
   * @param passiveData The passive dataset to score
   * @param passiveDataRandomEffectIds The set of random effect ids with passive data
   * @param modelsRDD The per-entity models
   * @return The computed scores for passive data
   */
  private def computePassiveScores(
      passiveData: RDD[(UniqueSampleId, (REId, LabeledPoint))],
      passiveDataRandomEffectIds: Broadcast[Set[REId]],
      modelsRDD: RDD[(REId, GeneralizedLinearModel)]): RDD[(Long, Double)] = {

    val modelsForPassiveData = modelsRDD
      .filter { case (shardId, _) =>
        passiveDataRandomEffectIds.value.contains(shardId)
      }
      .collectAsMap()

    val modelsForPassiveDataBroadcast = passiveData.sparkContext.broadcast(modelsForPassiveData)
    val passiveScores = passiveData.mapValues { case (randomEffectId, labeledPoint) =>
      modelsForPassiveDataBroadcast.value(randomEffectId).computeScore(labeledPoint.features)
    }

    // TODO: Need a better design that properly unpersists the broadcast variables and persists the computed RDD
//    passiveScores.setName("Passive scores").persist(StorageLevel.DISK_ONLY).count()
//    modelsForPassiveDataBroadcast.unpersist()

    passiveScores
  }
}
