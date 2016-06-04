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
import com.linkedin.photon.ml.constants.{MathConst, StorageLevel}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.model._
import com.linkedin.photon.ml.normalization.NoNormalization
import com.linkedin.photon.ml.optimization.GeneralizedLinearOptimizationProblem
import com.linkedin.photon.ml.optimization.game._
import com.linkedin.photon.ml.projector.{ProjectionMatrix, ProjectionMatrixBroadcast}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.VectorUtils
import org.apache.spark.rdd.RDD

/**
  * The optimization problem coordinate for a factored random effect model
  *
  * @param randomEffectDataSet The training dataset
  * @param factoredRandomEffectOptimizationProblem The fixed effect optimization problem
  */
protected[ml] class FactoredRandomEffectCoordinate[GLM <: GeneralizedLinearModel, F <: DiffFunction[LabeledPoint]](
    randomEffectDataSet: RandomEffectDataSet,
    factoredRandomEffectOptimizationProblem: FactoredRandomEffectOptimizationProblem[GLM, F])
  extends Coordinate[RandomEffectDataSet, FactoredRandomEffectCoordinate[GLM, F]](randomEffectDataSet) {

  /**
    * Initialize the model
    *
    * @param seed Random seed
    */
  protected[algorithm] override def initializeModel(seed: Long): DatumScoringModel = {
    val latentSpaceDimension = factoredRandomEffectOptimizationProblem.latentSpaceDimension
    FactoredRandomEffectCoordinate.initializeModel(
      randomEffectDataSet,
      factoredRandomEffectOptimizationProblem,
      latentSpaceDimension,
      seed)
  }

  /**
    * Update the model (i.e. run the coordinate optimizer)
    *
    * @param model Rhe model
    * @return Tuple of updated model and optimization tracker
    */
  protected[algorithm] override def updateModel(model: DatumScoringModel): (DatumScoringModel, OptimizationTracker) =
    model match {
      case factoredRandomEffectModel: FactoredRandomEffectModel =>
        var updatedModelsRDD = factoredRandomEffectModel.modelsInProjectedSpaceRDD
        var updatedProjectionMatrixBroadcast = factoredRandomEffectModel.projectionMatrixBroadcast
        var latentFactorOptimizationProblem = factoredRandomEffectOptimizationProblem.latentFactorOptimizationProblem

        val numIterations = factoredRandomEffectOptimizationProblem.numIterations
        val factoredRandomEffectOptimizationTracker =
          new Array[(RandomEffectOptimizationTracker, FixedEffectOptimizationTracker)](numIterations)
        val randomEffectOptimizationProblem = factoredRandomEffectOptimizationProblem.randomEffectOptimizationProblem
        val sparkContext = randomEffectDataSet.sparkContext

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
              .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
          randomEffectOptimizationTracker
              .setName(s"Random effect optimization tracker in iteration $iteration")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
          updatedRandomEffectModel.materialize()
          updatedModelsRDD.unpersist()
          updatedModelsRDD = updatedRandomEffectModel.modelsRDD

          // Then update the latent projection matrix
          val latentProjectionMatrix = updatedProjectionMatrixBroadcast.projectionMatrix
          val (updatedLatentProjectionMatrix, updatedLatentOptimizationProblem) =
            FactoredRandomEffectCoordinate.updateLatentProjectionMatrix(
              randomEffectDataSet,
              updatedRandomEffectModel,
              latentProjectionMatrix,
              latentFactorOptimizationProblem)

          updatedProjectionMatrixBroadcast.unpersistBroadcast()
          updatedProjectionMatrixBroadcast =
            new ProjectionMatrixBroadcast(sparkContext.broadcast(updatedLatentProjectionMatrix))

          // Note that the optimizationProblem will memorize the current state of optimization,
          // and the next round of updating latent factors will share the same convergence criteria as previous one.
          latentFactorOptimizationProblem = updatedLatentOptimizationProblem

          factoredRandomEffectOptimizationTracker(iteration) = (randomEffectOptimizationTracker,
            new FixedEffectOptimizationTracker(latentFactorOptimizationProblem.getStatesTracker.get))
        }

        // Return the updated model
        val updatedFactoredRandomEffectModel = factoredRandomEffectModel.updateFactoredRandomEffectModel(
          updatedModelsRDD,
          updatedProjectionMatrixBroadcast)

        (updatedFactoredRandomEffectModel,
          new FactoredRandomEffectOptimizationTracker(factoredRandomEffectOptimizationTracker))

      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} " +
            s"in ${this.getClass} is not supported!")
  }

  /**
    * Score the model
    *
    * @param model The model to score
    * @return Scores
    */
  protected[algorithm] override def score(model: DatumScoringModel): KeyValueScore = model match {
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
    * Compute the regularization term value
    *
    * @param model The model
    * @return Regularization term value
    */
  protected[algorithm] override def computeRegularizationTermValue(model: DatumScoringModel): Double = model match {
    case factoredRandomEffectModel: FactoredRandomEffectModel =>
      val modelsRDD = factoredRandomEffectModel.modelsInProjectedSpaceRDD
      val projectionMatrix = factoredRandomEffectModel.projectionMatrixBroadcast.projectionMatrix
      factoredRandomEffectOptimizationProblem.getRegularizationTermValue(modelsRDD, projectionMatrix)

    case _ =>
      throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
        s"type ${model.getClass} in ${this.getClass} is not supported!")
  }

  /**
    * Update the coordinate with a dataset
    *
    * @param updatedRandomEffectDataSet The updated dataset
    * @return The updated coordinate
    */
  override protected[algorithm] def updateCoordinateWithDataSet(updatedRandomEffectDataSet: RandomEffectDataSet)
    : FactoredRandomEffectCoordinate[GLM, F] =
    new FactoredRandomEffectCoordinate(updatedRandomEffectDataSet, factoredRandomEffectOptimizationProblem)
}


object FactoredRandomEffectCoordinate {
  /**
    * Initialize the model
    *
    * @param randomEffectDataSet The training dataset
    * @param latentSpaceDimension Dimensionality of the latent space
    * @param seed Random seed
    */
  private def initializeModel[GLM <: GeneralizedLinearModel, F <: DiffFunction[LabeledPoint]](
      randomEffectDataSet: RandomEffectDataSet,
      factoredRandomEffectOptimizationProblem: FactoredRandomEffectOptimizationProblem[GLM, F],
      latentSpaceDimension: Int,
      seed: Long): FactoredRandomEffectModel = {

    val randomEffectModelsRDD = randomEffectDataSet.activeData.mapValues(localDataSet =>
      factoredRandomEffectOptimizationProblem.initializeModel(latentSpaceDimension).asInstanceOf[GeneralizedLinearModel]
    )
    val numCols = latentSpaceDimension
    val latentProjectionMatrix = ProjectionMatrixBroadcast.buildRandomProjectionBroadcastProjector(
      randomEffectDataSet,
      numCols,
      isKeepingInterceptTerm = false,
      seed)
    val randomEffectId = randomEffectDataSet.randomEffectId
    val featureShardId = randomEffectDataSet.featureShardId
    new FactoredRandomEffectModel(randomEffectModelsRDD, latentProjectionMatrix, randomEffectId, featureShardId)
  }

  /**
    * Update the latent projection matrix
    *
    * @param randomEffectDataSet The dataset
    * @param randomEffectModel The model
    * @param projectionMatrix The projection matrix
    * @param latentFactorOptimizationProblem The optimization problem
    * @return Updated projection matrix
    */
  private def updateLatentProjectionMatrix[GLM <: GeneralizedLinearModel, F <: DiffFunction[LabeledPoint]](
      randomEffectDataSet: RandomEffectDataSet,
      randomEffectModel: RandomEffectModel,
      projectionMatrix: ProjectionMatrix,
      latentFactorOptimizationProblem: GeneralizedLinearOptimizationProblem[GLM, F])
    : (ProjectionMatrix, GeneralizedLinearOptimizationProblem[GLM, F]) = {

    val localDataSetRDD = randomEffectDataSet.activeData
    val modelsRDD = randomEffectModel.modelsRDD
    val latentProjectionMatrix = projectionMatrix.matrix
    val numRows = latentProjectionMatrix.rows
    val numCols = latentProjectionMatrix.cols
    val globalIdPartitioner = randomEffectDataSet.globalIdPartitioner
    val generatedTrainingData = kroneckerProductFeaturesAndCoefficients(
      localDataSetRDD,
      modelsRDD,
      sparsityToleranceThreshold = MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD)
    val downSampledTrainingData = latentFactorOptimizationProblem
      .downSample(generatedTrainingData)
      .partitionBy(globalIdPartitioner)
      .setName("Generated training data for latent projection matrix")
      .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val flattenedLatentProjectionMatrix = latentProjectionMatrix.flatten()
    val latentProjectionMatrixAsModel = latentFactorOptimizationProblem
      .initializeZeroModel(1)
      .updateCoefficients(Coefficients(flattenedLatentProjectionMatrix, variancesOption = None))
    val updatedModel = latentFactorOptimizationProblem.run(
      downSampledTrainingData.values,
      latentProjectionMatrixAsModel,
      NoNormalization)

    downSampledTrainingData.unpersist()

    val updatedLatentProjectionMatrix = Matrix.create(numRows, numCols, updatedModel.coefficients.means.toArray)
    (new ProjectionMatrix(updatedLatentProjectionMatrix), latentFactorOptimizationProblem)
  }

  /**
    * Computes the kronecker product between the dataset's features and the coefficients. Here the kronercker product is
    * defined as in [[https://en.wikipedia.org/wiki/Kronecker_product]], which is sometimes used interchangeably with
    * the terminology "cross product" or "outer product".
    *
    * @param localDataSetRDD The dataset
    * @param modelsRDD The coefficients
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
      localDataSet.dataPoints.map { case (globalId, labeledPoint) =>
        val generatedFeatures = VectorUtils.kroneckerProduct(
          labeledPoint.features,
          model.coefficients.means,
          sparsityToleranceThreshold)
        val generatedLabeledPoint =
          new LabeledPoint(labeledPoint.label, generatedFeatures, labeledPoint.offset, labeledPoint.weight)

        (globalId, generatedLabeledPoint)
      }
    }
  }
}
