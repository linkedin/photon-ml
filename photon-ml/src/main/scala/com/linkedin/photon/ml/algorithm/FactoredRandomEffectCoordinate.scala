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
package com.linkedin.photon.ml.algorithm

import breeze.linalg.Matrix
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.{MathConst, StorageLevel}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.function.{DistributedObjectiveFunction, SingleNodeObjectiveFunction}
import com.linkedin.photon.ml.model._
import com.linkedin.photon.ml.optimization.DistributedOptimizationProblem
import com.linkedin.photon.ml.optimization.game._
import com.linkedin.photon.ml.projector.{ProjectionMatrix, ProjectionMatrixBroadcast}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.VectorUtils

/**
 * The optimization problem coordinate for a random effect model using matrix factorization
 *
 * @tparam RandomEffectObjective The type of objective function used to solve individual random effect optimization
 *                               problems
 * @tparam LatentEffectObjective The type of objective function used to solve the latent factors optimization problem
 * @param randomEffectDataSet The training dataset
 * @param optimizationProblem The factored random effect optimization problem
 */
protected[ml] class FactoredRandomEffectCoordinate[
    RandomEffectObjective <: SingleNodeObjectiveFunction,
    LatentEffectObjective <: DistributedObjectiveFunction](
    randomEffectDataSet: RandomEffectDataSet,
    optimizationProblem: FactoredRandomEffectOptimizationProblem[RandomEffectObjective, LatentEffectObjective])
  extends Coordinate[
    RandomEffectDataSet,
    FactoredRandomEffectCoordinate[RandomEffectObjective, LatentEffectObjective]](
    randomEffectDataSet) {

  /**
   * Score the effect-specific data set in the coordinate with the input model
   *
   * @param model The input model
   * @return The output scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): KeyValueScore = model match {
    case factoredRandomEffectModel: FactoredRandomEffectModel =>
      val projectionMatrixBroadcast = factoredRandomEffectModel.projectionMatrixBroadcast
      val randomEffectModel = factoredRandomEffectModel.toRandomEffectModel
      val randomEffectDataSetInProjectedSpace = projectionMatrixBroadcast.projectRandomEffectDataSet(randomEffectDataSet)
      RandomEffectCoordinate.score(randomEffectDataSetInProjectedSpace, randomEffectModel)

    case _ =>
      throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
        s"in ${this.getClass} is not supported!")
  }

  /**
   * Initialize a basic model for scoring GAME data
   *
   * @param seed A random seed
   * @return The basic model
   */
  override protected[algorithm] def initializeModel(seed: Long): DatumScoringModel = {
    val latentSpaceDimension = optimizationProblem.latentSpaceDimension
    FactoredRandomEffectCoordinate.initializeModel(
      randomEffectDataSet,
      optimizationProblem,
      latentSpaceDimension,
      seed)
  }

  /**
   * Update the coordinate with a new dataset
   *
   * @param dataSet The updated dataset
   * @return A new coordinate with the updated dataset
   */
  override protected[algorithm] def updateCoordinateWithDataSet(
    dataSet: RandomEffectDataSet): FactoredRandomEffectCoordinate[RandomEffectObjective, LatentEffectObjective] =
    new FactoredRandomEffectCoordinate(dataSet, optimizationProblem)

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point
   *
   * @param model The model to use as a starting point
   * @return A tuple of the updated model and the optimization states tracker
   */
  override protected[algorithm] def updateModel(
    model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker]) = model match {
      case factoredRandomEffectModel: FactoredRandomEffectModel =>
        val sparkContext = randomEffectDataSet.sparkContext
        val randomEffectOptimizationProblem = optimizationProblem.randomEffectOptimizationProblem
        val latentFactorOptimizationProblem = optimizationProblem.latentFactorOptimizationProblem
        val isTrackingState = randomEffectOptimizationProblem.isTrackingState
        val numIterations = optimizationProblem.numIterations
        val factoredRandomEffectOptimizationTrackerArray =
          new Array[(RandomEffectOptimizationTracker, FixedEffectOptimizationTracker)](numIterations)

        var updatedModelsRDD = factoredRandomEffectModel.modelsInProjectedSpaceRDD
        var updatedProjectionMatrixBroadcast = factoredRandomEffectModel.projectionMatrixBroadcast

        for (iteration <- 0 until numIterations) {
          // First update the coefficients
          val randomEffectDataSetInProjectedSpace = updatedProjectionMatrixBroadcast.projectRandomEffectDataSet(
            randomEffectDataSet)
          val randomEffectModel = factoredRandomEffectModel
            .updateRandomEffectModelInProjectedSpace(updatedModelsRDD)
            .toRandomEffectModel
          val (updatedRandomEffectModel, randomEffectOptimizationTracker) = RandomEffectCoordinate.updateModel(
            randomEffectDataSetInProjectedSpace,
            randomEffectOptimizationProblem,
            randomEffectModel)

          updatedRandomEffectModel
            .modelsRDD
            .setName(s"Updated random effect model in iteration $iteration")
          updatedModelsRDD.unpersist()
          updatedModelsRDD = updatedRandomEffectModel.modelsRDD

          // Then update the latent projection matrix
          val latentProjectionMatrix = updatedProjectionMatrixBroadcast.projectionMatrix
          val updatedLatentProjectionMatrix = FactoredRandomEffectCoordinate.updateLatentProjectionMatrix(
            randomEffectDataSet,
            updatedRandomEffectModel,
            latentProjectionMatrix,
            latentFactorOptimizationProblem)

          updatedProjectionMatrixBroadcast.unpersistBroadcast()
          updatedProjectionMatrixBroadcast =
            new ProjectionMatrixBroadcast(sparkContext.broadcast(updatedLatentProjectionMatrix))

          if (isTrackingState) {
            randomEffectOptimizationTracker.get.setName(s"Random effect optimization tracker in iteration $iteration")
            factoredRandomEffectOptimizationTrackerArray(iteration) = (
              randomEffectOptimizationTracker.get,
              new FixedEffectOptimizationTracker(latentFactorOptimizationProblem.getStatesTracker.get))
          }
        }

        // Return the updated model
        val updatedFactoredRandomEffectModel = factoredRandomEffectModel.updateFactoredRandomEffectModel(
          updatedModelsRDD,
          updatedProjectionMatrixBroadcast)
        val factoredRandomEffectOptimizationTracker = if (isTrackingState) {
          Some(new FactoredRandomEffectOptimizationTracker(factoredRandomEffectOptimizationTrackerArray))
        } else None

        (updatedFactoredRandomEffectModel, factoredRandomEffectOptimizationTracker)

      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} " +
            s"in ${this.getClass} is not supported!")
  }

  /**
   * Compute the regularization term value of the coordinate for a given model
   *
   * @param model The model
   * @return The regularization term value
   */
  override protected[algorithm] def computeRegularizationTermValue(model: DatumScoringModel): Double = model match {
    case factoredRandomEffectModel: FactoredRandomEffectModel =>
      val modelsRDD = factoredRandomEffectModel.modelsInProjectedSpaceRDD
      val projectionMatrix = factoredRandomEffectModel.projectionMatrixBroadcast.projectionMatrix
      optimizationProblem.getRegularizationTermValue(modelsRDD, projectionMatrix)

    case _ =>
      throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
        s"type ${model.getClass} in ${this.getClass} is not supported!")
  }
}

object FactoredRandomEffectCoordinate {
  /**
   * Initialize a basic factored random effect model
   *
   * @tparam RandomFunc The type of objective function used to solve individual random effect optimization problems
   * @tparam LatentFunc The type of objective function used to solve the latent factors optimization problem
   * @param randomEffectDataSet The training dataset
   * @param factoredRandomEffectOptimizationProblem The optimization problem to use for creating the underlying models
   * @param latentSpaceDimension The dimensionality of the latent space
   * @param seed A random seed
   * @return A factored random effect model for scoring GAME data
   */
  private def initializeModel[RandomFunc <: SingleNodeObjectiveFunction, LatentFunc <: DistributedObjectiveFunction](
      randomEffectDataSet: RandomEffectDataSet,
      factoredRandomEffectOptimizationProblem: FactoredRandomEffectOptimizationProblem[RandomFunc, LatentFunc],
      latentSpaceDimension: Int,
      seed: Long): FactoredRandomEffectModel = {

    val glm = factoredRandomEffectOptimizationProblem.initializeModel(0)
    val randomEffectModelsRDD = randomEffectDataSet.activeData.mapValues(localDataSet =>
      glm.updateCoefficients(Coefficients.initializeZeroCoefficients(localDataSet.numFeatures))
        .asInstanceOf[GeneralizedLinearModel]
    )
    val numCols = latentSpaceDimension
    val latentProjectionMatrix = ProjectionMatrixBroadcast.buildRandomProjectionBroadcastProjector(
      randomEffectDataSet,
      numCols,
      isKeepingInterceptTerm = false,
      seed)
    val randomEffectType = randomEffectDataSet.randomEffectType
    val featureShardId = randomEffectDataSet.featureShardId
    new FactoredRandomEffectModel(randomEffectModelsRDD, latentProjectionMatrix, randomEffectType, featureShardId)
  }

  /**
   * Update the latent projection matrix
   *
   * @tparam Function The type of objective function used to solve the latent factors optimization problem
   * @param randomEffectDataSet The training dataset
   * @param randomEffectModel The individual random effect models
   * @param projectionMatrix The current projection matrix to use as a starting point
   * @param latentFactorOptimizationProblem The optimization problem for the factorization matrix
   * @return The updated projection matrix
   */
  private def updateLatentProjectionMatrix[Function <: DistributedObjectiveFunction](
      randomEffectDataSet: RandomEffectDataSet,
      randomEffectModel: RandomEffectModel,
      projectionMatrix: ProjectionMatrix,
      latentFactorOptimizationProblem: DistributedOptimizationProblem[Function]): ProjectionMatrix = {

    val localDataSetRDD = randomEffectDataSet.activeData
    val modelsRDD = randomEffectModel.modelsRDD
    val latentProjectionMatrix = projectionMatrix.matrix
    val numRows = latentProjectionMatrix.rows
    val numCols = latentProjectionMatrix.cols
    val uniqueIdPartitioner = randomEffectDataSet.uniqueIdPartitioner
    val generatedTrainingData = kroneckerProductFeaturesAndCoefficients(
        localDataSetRDD,
        modelsRDD,
        sparsityToleranceThreshold = MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD)
      .partitionBy(uniqueIdPartitioner)
      .setName("Generated training data for latent projection matrix")
      .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val flattenedLatentProjectionMatrix = latentProjectionMatrix.flatten()
    val latentProjectionMatrixAsModel = latentFactorOptimizationProblem
      .initializeZeroModel(1)
      .updateCoefficients(Coefficients(flattenedLatentProjectionMatrix, variancesOption = None))
    val updatedModel = latentFactorOptimizationProblem.runWithSampling(generatedTrainingData, latentProjectionMatrixAsModel)
    val updatedLatentProjectionMatrix = Matrix.create(numRows, numCols, updatedModel.coefficients.means.toArray)

    generatedTrainingData.unpersist()

    new ProjectionMatrix(updatedLatentProjectionMatrix)
  }

  /**
   * Computes the kronecker product between the dataset's features and the coefficients. Here the Kronecker product is
   * defined as in [[https://en.wikipedia.org/wiki/Kronecker_product]], which is sometimes used interchangeably with
   * the terminology "cross product" or "outer product".
   *
   * @param localDataSetRDD The training dataset
   * @param modelsRDD The individual random effect models
   * @param sparsityToleranceThreshold If the product between a certain feature and coefficient is smaller than
   *                                   sparsityToleranceThreshold, then it will be stored as 0 for sparsity
   *                                   consideration.
   * @return Kronecker product result
   */
  private def kroneckerProductFeaturesAndCoefficients(
      localDataSetRDD: RDD[(String, LocalDataSet)],
      modelsRDD: RDD[(String, GeneralizedLinearModel)],
      sparsityToleranceThreshold: Double = 0.0): RDD[(Long, LabeledPoint)] = {

    localDataSetRDD.join(modelsRDD).flatMap { case (_, (localDataSet, model)) =>
      localDataSet.dataPoints.map { case (uniqueId, labeledPoint) =>
        val generatedFeatures = VectorUtils.kroneckerProduct(
          labeledPoint.features,
          model.coefficients.means,
          sparsityToleranceThreshold)
        val generatedLabeledPoint =
          new LabeledPoint(labeledPoint.label, generatedFeatures, labeledPoint.offset, labeledPoint.weight)

        (uniqueId, generatedLabeledPoint)
      }
    }
  }
}
