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
import com.linkedin.photon.ml.model.{Coefficients, DatumScoringModel, RandomEffectModel}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.{RandomEffectOptimizationConfiguration, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * The optimization problem coordinate for a random effect model.
 *
 * @tparam Objective The type of objective function used to solve individual random effect optimization problems
 * @param dataset The training dataset
 * @param optimizationProblem The random effect optimization problem
 */
protected[ml] class RandomEffectCoordinate[Objective <: SingleNodeObjectiveFunction](
    override protected val  dataset: RandomEffectDataset,
    protected val optimizationProblem: RandomEffectOptimizationProblem[Objective])
  extends Coordinate[RandomEffectDataset](dataset)
    with ModelProjection
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
  override protected[algorithm] def updateCoordinateWithDataset(
      dataset: RandomEffectDataset): RandomEffectCoordinate[Objective] =
    new RandomEffectCoordinate(dataset, optimizationProblem)


  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(): (DatumScoringModel, OptimizationTracker) = {

    val (newModel, optimizationTracker) = RandomEffectCoordinate.trainModel(dataset, optimizationProblem, None)

    (projectModelBackward(newModel), optimizationTracker)
  }

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(model: DatumScoringModel): (DatumScoringModel, OptimizationTracker) =

    model match {
      case randomEffectModel: RandomEffectModel =>
        val (newModel, optimizationTracker) = RandomEffectCoordinate.trainModel(
          dataset,
          optimizationProblem,
          Some(projectModelForward(randomEffectModel)))

        (projectModelBackward(newModel), optimizationTracker)

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
      RandomEffectCoordinate.score(dataset, projectModelForward(randomEffectModel))

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
   * Helper function to construct [[RandomEffectCoordinate]] objects.
   *
   * @tparam RandomEffectObjective The type of objective function used to solve individual random effect optimization
   *                               problems
   * @param randomEffectDataset The data on which to run the optimization algorithm
   * @param configuration The optimization problem configuration
   * @param objectiveFunctionFactory The objective function factory option
   * @param priorRandomEffectModelOpt The prior randomEffectModel option
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param varianceComputationType If and how coefficient variances should be computed
   * @param interceptIndexOpt The index of the intercept, if there is one
   * @return A new [[RandomEffectCoordinate]]
   */
  protected[ml] def apply[RandomEffectObjective <: SingleNodeObjectiveFunction](
      randomEffectDataset: RandomEffectDataset,
      configuration: RandomEffectOptimizationConfiguration,
      objectiveFunctionFactory: (Option[GeneralizedLinearModel], Option[Int]) => RandomEffectObjective,
      priorRandomEffectModelOpt: Option[RandomEffectModel],
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: NormalizationContext,
      varianceComputationType: VarianceComputationType = VarianceComputationType.NONE,
      interceptIndexOpt: Option[Int] = None): RandomEffectCoordinate[RandomEffectObjective] = {

    // Generate parameters of ProjectedRandomEffectCoordinate
    val randomEffectOptimizationProblem = RandomEffectOptimizationProblem(
      randomEffectDataset.projectors,
      configuration,
      objectiveFunctionFactory,
      priorRandomEffectModelOpt,
      glmConstructor,
      normalizationContext,
      varianceComputationType,
      interceptIndexOpt)

    new RandomEffectCoordinate(randomEffectDataset, randomEffectOptimizationProblem)
  }

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
      initialRandomEffectModelOpt: Option[RandomEffectModel]): (RandomEffectModel, RandomEffectOptimizationTracker) = {

    // All 3 RDDs involved in the joins below use the same partitioner

    // Optimization problems are created for each entity with a projector, and thus guaranteed to match active data
    // exactly (see RandomEffectDataset.apply)
    val dataAndOptimizationProblems = randomEffectDataset
      .activeData
      .join(randomEffectOptimizationProblem.optimizationProblems)

    // Outer join the models to data and optimization problems
    val (newModels, randomEffectOptimizationTracker) = initialRandomEffectModelOpt
      .map { randomEffectModel =>
        val modelsAndTrackers = randomEffectModel
          .modelsRDD
          .fullOuterJoin(dataAndOptimizationProblems)
          .mapValues {
            case (Some(localModel), Some((localDataset, optimizationProblem))) =>
              val trainingLabeledPoints = localDataset.dataPoints.map(_._2)
              val updatedModel = optimizationProblem.run(trainingLabeledPoints, localModel)
              val stateTrackers = optimizationProblem.getStatesTracker

              (updatedModel, Some(stateTrackers))

            case (Some(localModel), None) =>
              (localModel, None)

            case (None, Some((localDataset, optimizationProblem))) =>
              val trainingLabeledPoints = localDataset.dataPoints.map(_._2)
              val updatedModel = optimizationProblem.run(trainingLabeledPoints)
              val stateTrackers = optimizationProblem.getStatesTracker

              (updatedModel, Some(stateTrackers))

            case _ =>
              throw new IllegalStateException("Either a initial random effect model or data should be present!")
          }
        modelsAndTrackers.persist(StorageLevel.MEMORY_ONLY_SER)

        val models = modelsAndTrackers.mapValues(_._1)
        val optimizationTracker = RandomEffectOptimizationTracker(modelsAndTrackers.flatMap(_._2._2))

        (models, optimizationTracker)
      }
      .getOrElse {
        val modelsAndTrackers = dataAndOptimizationProblems.mapValues { case (localDataset, optimizationProblem) =>
          val trainingLabeledPoints = localDataset.dataPoints.map(_._2)
          val newModel = optimizationProblem.run(trainingLabeledPoints)
          val stateTrackers = optimizationProblem.getStatesTracker

          (newModel, stateTrackers)
        }
        modelsAndTrackers.persist(StorageLevel.MEMORY_ONLY_SER)

        val models = modelsAndTrackers.mapValues(_._1)
        val optimizationTracker = RandomEffectOptimizationTracker(modelsAndTrackers.map(_._2._2))

        (models, optimizationTracker)
      }

    val newRandomEffectModel = new RandomEffectModel(
      newModels,
      randomEffectDataset.randomEffectType,
      randomEffectDataset.featureShardId)

    (newRandomEffectModel, randomEffectOptimizationTracker)
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

    // There may be more models than active data. However, since we're computing residuals for future coordinates, no
    // data means no residual. Therefore, we use an inner join. Note that the active data and models use the same
    // partitioner, but scores need to use GameDatum partitioner.
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
