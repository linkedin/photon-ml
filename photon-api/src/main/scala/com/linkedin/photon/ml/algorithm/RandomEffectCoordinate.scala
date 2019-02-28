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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.{DatumScoringModel, RandomEffectModel}
import com.linkedin.photon.ml.optimization.game.RandomEffectOptimizationProblem
import com.linkedin.photon.ml.optimization.{OptimizationTracker, RandomEffectOptimizationTracker}
import com.linkedin.photon.ml.spark.RDDLike

/**
 * The optimization problem coordinate for a random effect model.
 *
 * @tparam Objective The type of objective function used to solve individual random effect optimization problems
 * @param dataset The training dataset
 * @param optimizationProblem The random effect optimization problem
 */
protected[ml] class RandomEffectCoordinate[Objective <: SingleNodeObjectiveFunction](
    protected val dataset: RandomEffectDataset,
    protected val optimizationProblem: RandomEffectOptimizationProblem[Objective])
  extends Coordinate[RandomEffectDataset](dataset)
    with RDDLike {

  //
  // Coordinate functions
  //

  /**
   * Update the coordinate with a new [[RandomEffectDataset]].
   *
   * @param dataset The updated [[RandomEffectDataset]]
   * @return A new coordinate with the updated [[RandomEffectDataset]]
   */
  override protected def updateCoordinateWithDataset(dataset: RandomEffectDataset): RandomEffectCoordinate[Objective] =
    new RandomEffectCoordinate(dataset, optimizationProblem)


  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(): (DatumScoringModel, Option[OptimizationTracker]) =
    RandomEffectCoordinate.trainModel(dataset, optimizationProblem, None)

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(
      model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker]) =
    model match {
      case randomEffectModel: RandomEffectModel =>
        RandomEffectCoordinate.trainModel(dataset, optimizationProblem, Some(randomEffectModel))

      case _ =>
        throw new UnsupportedOperationException(
          s"Updating model of type ${model.getClass} in ${this.getClass} is not supported")
    }

  /**
   * Compute scores for the coordinate data using a given model.
   *
   * @param model The input model
   * @return The dataset scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = model match {

    case randomEffectModel: RandomEffectModel =>
      RandomEffectCoordinate.score(dataset, randomEffectModel)

    case _ =>
      throw new UnsupportedOperationException(
        s"Scoring with model of type ${model.getClass} in ${this.getClass} is not supported")
  }

  //
  // RDDLike Functions
  //

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = optimizationProblem.sparkContext

  /**
   * Assign a given name to the [[optimizationProblem]] [[RDD]].
   *
   * @param name The parent name for all [[RDD]] objects in this class
   * @return This object with the name of the [[optimizationProblem]] [[RDD]] assigned
   */
  override def setName(name: String): RandomEffectCoordinate[Objective] = {

    optimizationProblem.setName(name)

    this
  }

  /**
   * Set the persistence storage level of the [[optimizationProblem]] [[RDD]].
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of the [[optimizationProblem]] [[RDD]] set
   */
  override def persistRDD(storageLevel: StorageLevel): RandomEffectCoordinate[Objective] = {

    optimizationProblem.persistRDD(storageLevel)

    this
  }

  /**
   * Mark the [[optimizationProblem]] [[RDD]] as unused, and asynchronously remove all blocks for it from memory and
   * disk.
   *
   * @return This object with the [[optimizationProblem]] [[RDD]] unpersisted
   */
  override def unpersistRDD(): RandomEffectCoordinate[Objective] = {

    optimizationProblem.unpersistRDD()

    this
  }

  /**
   * Materialize the [[optimizationProblem]] [[RDD]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be
   * evaluated).
   *
   * @return This object with the [[optimizationProblem]] [[RDD]] materialized
   */
  override def materialize(): RandomEffectCoordinate[Objective] = {

    optimizationProblem.materialize()

    this
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

    // All 3 RDDs involved in these joins use the same partitioner
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

        Some(RandomEffectOptimizationTracker(stateTrackers))

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
   * @note The score is the raw dot product of the model coefficients and the feature values - it does not go through a
   *       non-linear link function.
   * @param randomEffectDataset The [[RandomEffectDataset]] to score
   * @param randomEffectModel The [[RandomEffectModel]] with which to score
   * @return The computed scores
   */
  protected[algorithm] def score(
      randomEffectDataset: RandomEffectDataset,
      randomEffectModel: RandomEffectModel): CoordinateDataScores = {

    // Active data and models use the same partitioner, but scores need to use GameDatum partitioner
    val activeScores = randomEffectDataset
      .activeData
      .join(randomEffectModel.modelsRDD)
      .flatMap { case (_, (localDataset, model)) =>
        localDataset.dataPoints.map { case (uniqueId, labeledPoint) =>
          (uniqueId, model.computeScore(labeledPoint.features))
        }
      }
      .partitionBy(randomEffectDataset.uniqueIdPartitioner)

    // Passive data already uses the GameDatum partitioner. Note that this code assumes few (if any) entities have a
    // passive dataset.
    val passiveDataREIds = randomEffectDataset.passiveDataREIds
    val modelsForPassiveData = randomEffectModel
      .modelsRDD
      .filter { case (reId, _) =>
        passiveDataREIds.value.contains(reId)
      }
      .collectAsMap()
    val passiveScores = randomEffectDataset
      .passiveData
      .mapValues { case (randomEffectId, labeledPoint) =>
        modelsForPassiveData(randomEffectId).computeScore(labeledPoint.features)
      }

    new CoordinateDataScores(activeScores ++ passiveScores)
  }
}
