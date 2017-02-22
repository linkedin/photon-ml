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
import com.linkedin.photon.ml.sampler.DownSampler
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

  override def sparkContext: SparkContext = randomEffectOptimizationProblem.sparkContext

  override def setName(name: String): this.type = {
    randomEffectOptimizationProblem.setName(name)
    this
  }

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    randomEffectOptimizationProblem.persistRDD(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    randomEffectOptimizationProblem.unpersistRDD()
    this
  }

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
   * @param randomEffectDataSet The training data
   * @param randomEffectOptimizationConfiguration The optimizer configuration for the random effect optimization
   *                                              problems
   * @param latentFactorOptimizationConfiguration The optimizer configuration for the latent effect optimization problem
   * @param mfOptimizationConfiguration The configuration for the factorization matrix used to compute the latent
   *                                    effects
   * @param randomObjectiveFunction The objective function to optimize for the random effects
   * @param latentObjectiveFunction The objective function to optimize for the latent effects matrix
   * @param latentSamplerOption (Optional) A sampler to use for down-sampling the training data prior to optimization of
   *                            the latent effects matrix
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param isTrackingState Should the optimization problems record the internal optimizer states?
   * @param isComputingVariance Should coefficient variances be computed in addition to the means?
   * @return A new RandomEffectOptimizationProblem
   */
  protected[ml] def create[
      RandomEffectObjective <: SingleNodeObjectiveFunction,
      LatentEffectObjective <: DistributedObjectiveFunction](
      randomEffectDataSet: RandomEffectDataSet,
      randomEffectOptimizationConfiguration: GLMOptimizationConfiguration,
      latentFactorOptimizationConfiguration: GLMOptimizationConfiguration,
      mfOptimizationConfiguration: MFOptimizationConfiguration,
      randomObjectiveFunction: RandomEffectObjective,
      latentObjectiveFunction: LatentEffectObjective,
      latentSamplerOption: Option[DownSampler],
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: Broadcast[NormalizationContext],
      isTrackingState: Boolean = false,
      isComputingVariance: Boolean = false)
    : FactoredRandomEffectOptimizationProblem[RandomEffectObjective, LatentEffectObjective] = {

    val MFOptimizationConfiguration(numInnerIterations, latentSpaceDimension) = mfOptimizationConfiguration
    val randomEffectOptimizationProblem = RandomEffectOptimizationProblem.create(
      randomEffectDataSet,
      randomEffectOptimizationConfiguration,
      randomObjectiveFunction,
      glmConstructor,
      normalizationContext,
      isTrackingState,
      isComputingVariance)
    val latentFactorOptimizationProblem = DistributedOptimizationProblem.create(
      latentFactorOptimizationConfiguration,
      latentObjectiveFunction,
      latentSamplerOption,
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
