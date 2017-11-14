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
package com.linkedin.photon.ml.optimization.game

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.data.RandomEffectDataSet
import com.linkedin.photon.ml.function.{DistributedObjectiveFunction, SingleNodeObjectiveFunction}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.projector.ProjectionMatrix
import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Representation for a factored random effect optimization problem.
 *
 * @tparam RandomEffectObjective The objective function to optimize for the individual random effects
 * @tparam LatentEffectObjective The objective function to optimize for the single latent effects model
 * @param randomEffectOptimizationProblem The random effect optimization problem
 * @param latentFactorOptimizationProblem The latent factor optimization problem
 * @param numIterations The number of internal iterations to perform for refining the latent factor approximation
 * @param latentSpaceDimension The dimensionality of the latent space
 */
protected[ml] class FactoredRandomEffectOptimizationProblem[
    RandomEffectObjective <: SingleNodeObjectiveFunction,
    LatentEffectObjective <: DistributedObjectiveFunction](
    val randomEffectOptimizationProblem: RandomEffectOptimizationProblem[RandomEffectObjective],
    val latentFactorOptimizationProblem: DistributedOptimizationProblem[LatentEffectObjective],
    val numIterations: Int,
    val latentSpaceDimension: Int)
  extends RDDLike {

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = randomEffectOptimizationProblem.sparkContext

  /**
   * Assign a given name to [[randomEffectOptimizationProblem]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   *
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the name of [[randomEffectOptimizationProblem]] assigned
   */
  override def setName(name: String): this.type = {
    randomEffectOptimizationProblem.setName(name)
    this
  }

  /**
   * Set the storage level of [[randomEffectOptimizationProblem]], and persist its values across the cluster the first
   * time they are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[randomEffectOptimizationProblem]] set
   */
  override def persistRDD(storageLevel: StorageLevel): this.type = {
    randomEffectOptimizationProblem.persistRDD(storageLevel)
    this
  }

  /**
   * Mark [[randomEffectOptimizationProblem]] as non-persistent, and remove all blocks for it from memory and disk.
   *
   * @return This object with [[randomEffectOptimizationProblem]] marked non-persistent
   */
  override def unpersistRDD(): this.type = {
    randomEffectOptimizationProblem.unpersistRDD()
    this
  }

  /**
   * Materialize [[randomEffectOptimizationProblem]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be
   * evaluated).
   *
   * @return This object with [[randomEffectOptimizationProblem]] materialized
   */
  override def materialize(): this.type = {
    randomEffectOptimizationProblem.materialize()
    this
  }

  /**
   * Create a default generalized linear model with 0-valued coefficients
   *
   * @param dimension The dimensionality of the model coefficients
   * @return A model with zero coefficients
   */
  def initializeModel(dimension: Int): GeneralizedLinearModel =
    latentFactorOptimizationProblem.initializeZeroModel(dimension)

  /**
   * Compute the regularization term value
   *
   * @param modelsRDD The coefficients
   * @param projectionMatrix The projection matrix
   * @return Regularization term value
   */
  def getRegularizationTermValue(modelsRDD: RDD[(String, GeneralizedLinearModel)], projectionMatrix: ProjectionMatrix)
    : Double = {

    val projectionMatrixAsCoefficients = new Coefficients(projectionMatrix.matrix.flatten(), variancesOption = None)
    val projectionMatrixModel = latentFactorOptimizationProblem
      .initializeZeroModel(1)
      .updateCoefficients(projectionMatrixAsCoefficients)

    randomEffectOptimizationProblem.getRegularizationTermValue(modelsRDD) +
        latentFactorOptimizationProblem.getRegularizationTermValue(projectionMatrixModel)
  }
}

object FactoredRandomEffectOptimizationProblem {
  /**
   * Factory method to create new RandomEffectOptimizationProblems.
   *
   * @tparam RandomEffectObjective The objective function type of the random effects
   * @tparam LatentEffectObjective The objective function type of the latent effects matrix
   * @param randomEffectDataSet The training data
   * @param randomEffectOptimizationConfiguration The optimizer configuration for the random effect optimization
   *                                              problems
   * @param latentFactorOptimizationConfiguration The optimizer configuration for the latent effect optimization problem
   * @param mfOptimizationConfiguration The configuration for the factorization matrix used to compute the latent
   *                                    effects
   * @param randomObjectiveFunction The objective function to optimize for the random effects
   * @param latentObjectiveFunction The objective function to optimize for the latent effects matrix
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param isTrackingState Should the optimization problems record the internal optimizer states?
   * @param isComputingVariance Should coefficient variances be computed in addition to the means?
   * @return A new RandomEffectOptimizationProblem
   */
  protected[ml] def apply[
      RandomEffectObjective <: SingleNodeObjectiveFunction,
      LatentEffectObjective <: DistributedObjectiveFunction](
      randomEffectDataSet: RandomEffectDataSet,
      randomEffectOptimizationConfiguration: GLMOptimizationConfiguration,
      latentFactorOptimizationConfiguration: GLMOptimizationConfiguration,
      mfOptimizationConfiguration: MFOptimizationConfiguration,
      randomObjectiveFunction: RandomEffectObjective,
      latentObjectiveFunction: LatentEffectObjective,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: Broadcast[NormalizationContext],
      isTrackingState: Boolean = false,
      isComputingVariance: Boolean = false)
    : FactoredRandomEffectOptimizationProblem[RandomEffectObjective, LatentEffectObjective] = {

    val MFOptimizationConfiguration(numInnerIterations, latentSpaceDimension) = mfOptimizationConfiguration
    val randomEffectOptimizationProblem = RandomEffectOptimizationProblem(
      randomEffectDataSet,
      randomEffectOptimizationConfiguration,
      randomObjectiveFunction,
      glmConstructor,
      normalizationContext,
      isTrackingState,
      isComputingVariance)
    val latentFactorOptimizationProblem = DistributedOptimizationProblem(
      latentFactorOptimizationConfiguration,
      latentObjectiveFunction,
      samplerOption = None,
      glmConstructor,
      normalizationContext,
      isTrackingState,
      // Don't want to compute variance of the projection matrix
      isComputingVariance = false)

    new FactoredRandomEffectOptimizationProblem(
      randomEffectOptimizationProblem,
      latentFactorOptimizationProblem,
      numInnerIterations,
      latentSpaceDimension)
  }
}
