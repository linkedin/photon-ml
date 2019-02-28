/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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

import scala.collection.mutable

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.data.RandomEffectDataset
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.{Coefficients, DatumScoringModel, RandomEffectModel}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.optimization.{RandomEffectOptimizationTracker, SingleNodeOptimizationProblem, VarianceComputationType}
import com.linkedin.photon.ml.optimization.game.{RandomEffectOptimizationConfiguration, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.projector.dataset.{LinearSubspaceREDProjector, MatrixREDProjector}
import com.linkedin.photon.ml.projector.{LinearSubspaceProjection, RandomProjection}
import com.linkedin.photon.ml.projector.model.{LinearSubspaceREMProjector, MatrixREMProjector, RandomEffectModelProjector}
import com.linkedin.photon.ml.projector.vector.{LinearSubspaceProjector, MatrixProjector}
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.{PhotonNonBroadcast, VectorUtils}

/**
 * The optimization problem coordinate for a random effect model in projected space.
 *
 * @tparam Objective The type of objective function used to solve individual random effect optimization problems
 * @param dataset The training dataset
 * @param optimizationProblem The fixed effect optimization problem
 */
protected[ml] class ProjectedRandomEffectCoordinate[Objective <: SingleNodeObjectiveFunction](
    override protected val dataset: RandomEffectDataset,
    override protected val optimizationProblem: RandomEffectOptimizationProblem[Objective],
    randomEffectModelProjector: RandomEffectModelProjector)
  extends RandomEffectCoordinate[Objective](dataset, optimizationProblem)
    with BroadcastLike {

  //
  // Coordinate Functions
  //

  /**
   * Compute scores for the coordinate data using a given model.
   *
   * @param model The input model
   * @return The dataset scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = model match {

    case randomEffectModel: RandomEffectModel =>
      val projectedModel = randomEffectModelProjector.projectForward(randomEffectModel).persistRDD(StorageLevel.MEMORY_ONLY)
      val result = super.score(projectedModel)

      projectedModel.unpersistRDD()

      result

    case _ =>
      throw new UnsupportedOperationException(
        s"Scoring with model of type ${model.getClass} in ${this.getClass} is not supported")
  }

  /**
   * Initialize a [[RandomEffectModel]] with all-0 coefficients.
   *
   * @param seed A random seed
   * @return The basic model
   */
  override protected[ml] def initializeModel(seed: Long): RandomEffectModel =
    randomEffectModelProjector.projectBackward(super.initializeModel(seed))

  /**
   * Update the coordinate with a new [[RandomEffectDataset]].
   *
   * @param dataset The updated [[RandomEffectDataset]]
   * @return A new coordinate with the updated [[RandomEffectDataset]]
   */
  override protected def updateCoordinateWithDataset(
      dataset: RandomEffectDataset): ProjectedRandomEffectCoordinate[Objective] =
    new ProjectedRandomEffectCoordinate(dataset, optimizationProblem, randomEffectModelProjector)

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def updateModel(
      model: DatumScoringModel): (RandomEffectModel, Option[RandomEffectOptimizationTracker]) =
    model match {
      case randomEffectModel: RandomEffectModel =>
        val (newModel, optimizationTrackerOpt) =
          super.updateModel(randomEffectModelProjector.projectForward(randomEffectModel))

        (randomEffectModelProjector.projectBackward(newModel), optimizationTrackerOpt)

      case _ =>
        throw new UnsupportedOperationException(
          s"Updating model of type ${model.getClass} in ${this.getClass} is not supported")
    }

  //
  // BroadcastLike Functions
  //

  /**
   * Asynchronously delete cached copies of the [[randomEffectModelProjector]] on all executors, if it is broadcast.
   *
   * @return This [[RandomEffectDataset]] with [[randomEffectModelProjector]] unpersisted (if it was broadcast)
   */
  override protected[ml] def unpersistBroadcast(): ProjectedRandomEffectCoordinate[Objective] = {

    randomEffectModelProjector match {
      case broadcastLike: BroadcastLike => broadcastLike.unpersistBroadcast()
      case _ =>
    }

    this
  }

  //
  // RDDLike Functions
  //

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = super.sparkContext

  /**
   * Assign a given name to the [[randomEffectModelProjector]] [[RDD]] (if it is a [[RDD]]) and all superclass [[RDD]]
   * objects.
   *
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the names of the [[randomEffectModelProjector]] [[RDD]] (if it is a [[RDD]]) and all
   *         superclass [[RDD]] objects assigned
   */
  override def setName(name: String): ProjectedRandomEffectCoordinate[Objective] = {

    super.setName(name)

    randomEffectModelProjector match {
      case rddLike: RDDLike => rddLike.setName(name)
      case _ =>
    }

    this
  }

  /**
   * Set the persistence storage level of the [[randomEffectModelProjector]] [[RDD]] (if it is a [[RDD]]) and all
   * superclass [[RDD]] objects.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of the [[randomEffectModelProjector]] [[RDD]] (if it is a [[RDD]]) and
   *         all superclass [[RDD]] objects set
   */
  override def persistRDD(storageLevel: StorageLevel): ProjectedRandomEffectCoordinate[Objective] = {

    super.persistRDD(storageLevel)

    randomEffectModelProjector match {
      case rddLike: RDDLike => rddLike.persistRDD(storageLevel)
      case _ =>
    }

    this
  }

  /**
   * Mark the [[randomEffectModelProjector]] [[RDD]] (if it is a [[RDD]]) and all superclass [[RDD]] objects as unused,
   * and asynchronously remove all blocks for it from memory and disk.
   *
   * @return This object with the [[randomEffectModelProjector]] [[RDD]] (if it is a [[RDD]]) and all superclass [[RDD]]
   *         objects unpersisted
   */
  override def unpersistRDD(): ProjectedRandomEffectCoordinate[Objective] = {

    super.unpersistRDD()

    randomEffectModelProjector match {
      case rddLike: RDDLike => rddLike.unpersistRDD()
      case _ =>
    }

    this
  }

  /**
   * Materialize the [[randomEffectModelProjector]] [[RDD]] (if it is a [[RDD]]) and all superclass [[RDD]] objects
   * (Spark [[RDD]]s are lazy evaluated: this method forces them to be evaluated).
   *
   * @return This object with the [[randomEffectModelProjector]] [[RDD]] (if it is a [[RDD]]) and all superclass [[RDD]]
   *         objects materialized
   */
  override def materialize(): ProjectedRandomEffectCoordinate[Objective] = {

    super.materialize()

    randomEffectModelProjector match {
      case rddLike: RDDLike => rddLike.materialize()
      case _ =>
    }

    this
  }
}

object ProjectedRandomEffectCoordinate {

  /**
   *
   *
   * @param randomEffectDataset
   * @param configuration
   * @param objectiveFunction
   * @param glmConstructor
   * @param normalizationContext
   * @param varianceComputationType
   * @param isTrackingState
   * @tparam RandomEffectObjective
   * @return
   */
  protected[ml] def apply[RandomEffectObjective <: SingleNodeObjectiveFunction](
      randomEffectDataset: RandomEffectDataset,
      configuration: RandomEffectOptimizationConfiguration,
      objectiveFunction: RandomEffectObjective,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: NormalizationContext,
      varianceComputationType: VarianceComputationType = VarianceComputationType.NONE,
      isTrackingState: Boolean = false): ProjectedRandomEffectCoordinate[RandomEffectObjective] = {

    val originalSpaceDimension = randomEffectDataset.activeData.take(1).head._2.numFeatures

    val (projectedDataset, optimizationProblem, modelProjector) = configuration.projectionType match {
      case LinearSubspaceProjection =>

        // Build projectors for each entity
        val linearSubspaceProjectorsRDD = buildLinearSubspaceProjectors(randomEffectDataset, originalSpaceDimension)

        // Generate parameters of ProjectedRandomEffectCoordinate
        val projectedRandomEffectDataset =
          new LinearSubspaceREDProjector(linearSubspaceProjectorsRDD).projectForward(randomEffectDataset)
        val randomEffectOptimizationProblem = buildLinearSubspaceProjectionRandomEffectOptimizationProblem(
          linearSubspaceProjectorsRDD,
          configuration,
          objectiveFunction,
          glmConstructor,
          normalizationContext,
          varianceComputationType,
          isTrackingState)
        val linearSubspaceREMProjector = new LinearSubspaceREMProjector(linearSubspaceProjectorsRDD)

        (projectedRandomEffectDataset, randomEffectOptimizationProblem, linearSubspaceREMProjector)

      case RandomProjection(projectedSpaceDimension) =>
        val randomMatrixProjectorBroadcast = buildMatrixProjector(
          randomEffectDataset,
          originalSpaceDimension,
          projectedSpaceDimension)

        // Generate parameters of ProjectedRandomEffectCoordinate
        val projectedRandomEffectDataset =
          new MatrixREDProjector(randomMatrixProjectorBroadcast).projectForward(randomEffectDataset)
        val randomEffectOptimizationProblem = buildMatrixProjectionRandomEffectOptimizationProblem(
          randomMatrixProjectorBroadcast,
          randomEffectDataset,
          configuration,
          objectiveFunction,
          glmConstructor,
          normalizationContext,
          varianceComputationType,
          isTrackingState)
        val matrixREMProjector = new MatrixREMProjector(randomMatrixProjectorBroadcast)

        (projectedRandomEffectDataset, randomEffectOptimizationProblem, matrixREMProjector)

      case p =>
        throw new UnsupportedOperationException(s"Unsupported projection type '$p' encountered")
    }

    projectedDataset.persistRDD(StorageLevel.DISK_ONLY_2)
    optimizationProblem.persistRDD(StorageLevel.DISK_ONLY)

    new ProjectedRandomEffectCoordinate(
      projectedDataset,
      optimizationProblem,
      modelProjector)
  }

  /**
   *
   * @param randomEffectDataset
   * @param originalSpaceDimension
   * @return
   */
  private def buildLinearSubspaceProjectors(
      randomEffectDataset: RandomEffectDataset,
      originalSpaceDimension: Int): RDD[(REId, LinearSubspaceProjector)] = {

    // Collect active indices for the active data set
    val activeDataIndices = randomEffectDataset
      .activeData
      .mapValues { localDataset =>
        localDataset
          .dataPoints
          .foldLeft(mutable.Set[Int]()) { case (indices, (_, labeledPoint)) =>
            indices ++ VectorUtils.getActiveIndices(labeledPoint.features)
          }
          .toSet
      }

    // Collect active indices for the passive data set
    val passiveDataIndices = randomEffectDataset
      .passiveData
      .map { case (_, (reId, labeledPoint)) =>
        (reId, VectorUtils.getActiveIndices(labeledPoint.features))
      }
      .partitionBy(randomEffectDataset.randomEffectIdPartitioner)

    // Union them, and fold the results into (reId, indices) tuples
    val activeIndicesRdd = activeDataIndices
      .union(passiveDataIndices)
      .foldByKey(Set.empty[Int])(_ ++ _)

    // Generate projectors from indices
    activeIndicesRdd
      .mapValues { activeIndices =>
        new LinearSubspaceProjector(activeIndices, originalSpaceDimension)
      }
      .setName(s"${randomEffectDataset.randomEffectType} - LinearSpaceProjectors")
      .persist(StorageLevel.DISK_ONLY)
  }

  /**
   *
   * @param linearSubspaceProjectorsRDD
   * @param configuration
   * @param objectiveFunction
   * @param glmConstructor
   * @param normalizationContext
   * @param varianceComputationType
   * @param isTrackingState
   * @tparam RandomEffectObjective
   * @return
   */
  private def buildLinearSubspaceProjectionRandomEffectOptimizationProblem[RandomEffectObjective <: SingleNodeObjectiveFunction](
      linearSubspaceProjectorsRDD: RDD[(REId, LinearSubspaceProjector)],
      configuration: RandomEffectOptimizationConfiguration,
      objectiveFunction: RandomEffectObjective,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: NormalizationContext,
      varianceComputationType: VarianceComputationType = VarianceComputationType.NONE,
      isTrackingState: Boolean = false): RandomEffectOptimizationProblem[RandomEffectObjective] = {

    // Generate new NormalizationContext and SingleNodeOptimizationProblem objects
    val optimizationProblems = linearSubspaceProjectorsRDD
      .mapValues { projector =>
        val factors = normalizationContext.factorsOpt.map(factors => projector.projectForward(factors))
        val shiftsAndIntercept = normalizationContext
          .shiftsAndInterceptOpt
          .map { case (shifts, intercept) =>
            val newShifts = projector.projectForward(shifts)
            val newIntercept = projector.originalToProjectedSpaceMap(intercept)

            (newShifts, newIntercept)
          }
        val projectedNormalizationContext = new NormalizationContext(factors, shiftsAndIntercept)

        SingleNodeOptimizationProblem(
          configuration,
          objectiveFunction,
          glmConstructor,
          PhotonNonBroadcast(projectedNormalizationContext),
          varianceComputationType,
          isTrackingState)
      }

    new RandomEffectOptimizationProblem(optimizationProblems, glmConstructor, isTrackingState)
  }

  /**
   *
   * @param randomEffectDataset
   * @param originalSpaceDimension
   * @param projectedSpaceDimension
   * @return
   */
  private def buildMatrixProjector(
      randomEffectDataset: RandomEffectDataset,
      originalSpaceDimension: Int,
      projectedSpaceDimension: Int): Broadcast[MatrixProjector] =
    randomEffectDataset
      .sparkContext
      .broadcast(
        MatrixProjector.buildGaussianRandomMatrixProjector(
          originalSpaceDimension,
          projectedSpaceDimension,
          isKeepingInterceptTerm = true))

  /**
   *
   * @param randomMatrixProjectorBroadcast
   * @param randomEffectDataset
   * @param configuration
   * @param objectiveFunction
   * @param glmConstructor
   * @param normalizationContext
   * @param varianceComputationType
   * @param isTrackingState
   * @tparam RandomEffectObjective
   * @return
   */
  private def buildMatrixProjectionRandomEffectOptimizationProblem[RandomEffectObjective <: SingleNodeObjectiveFunction](
    randomMatrixProjectorBroadcast: Broadcast[MatrixProjector],
    randomEffectDataset: RandomEffectDataset,
    configuration: RandomEffectOptimizationConfiguration,
    objectiveFunction: RandomEffectObjective,
    glmConstructor: Coefficients => GeneralizedLinearModel,
    normalizationContext: NormalizationContext,
    varianceComputationType: VarianceComputationType = VarianceComputationType.NONE,
    isTrackingState: Boolean = false): RandomEffectOptimizationProblem[RandomEffectObjective] = {

    // Generate new NormalizationContext and SingleNodeOptimizationProblem objects
    val optimizationProblems = randomEffectDataset
      .activeData
      .mapValues { _ =>
        val projector = randomMatrixProjectorBroadcast.value

        val factors = normalizationContext
          .factorsOpt
          .map { factors =>
            projector.projectForward(factors)
          }
        val shiftsAndIntercept = normalizationContext
          .shiftsAndInterceptOpt
          .map { case (shifts, _) =>
            val newShifts = projector.projectForward(shifts)
            val newIntercept = projector.projectedInterceptId

            (newShifts, newIntercept)
          }
        val projectedNormalizationContext = new NormalizationContext(factors, shiftsAndIntercept)

        SingleNodeOptimizationProblem(
          configuration,
          objectiveFunction,
          glmConstructor,
          PhotonNonBroadcast(projectedNormalizationContext),
          varianceComputationType,
          isTrackingState)
      }

    new RandomEffectOptimizationProblem(optimizationProblems, glmConstructor, isTrackingState)
  }
}
