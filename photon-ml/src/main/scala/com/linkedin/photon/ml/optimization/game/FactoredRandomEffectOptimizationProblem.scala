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
package com.linkedin.photon.ml.optimization.game

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.data.{LabeledPoint, RandomEffectDataSet}
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.projector.ProjectionMatrix
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * An optimization problem for factored random effect datasets
  *
  * @param randomEffectOptimizationProblem The random effect optimization problem
  * @param latentFactorOptimizationProblem The latent factor optimization problem
  * @param numIterations The number of internal iterations to perform for refining the latent factor approximation
  * @param latentSpaceDimension The dimensionality of the latent space
  */
protected[ml] class FactoredRandomEffectOptimizationProblem[GLM <: GeneralizedLinearModel,
  F <: DiffFunction[LabeledPoint]](
    val randomEffectOptimizationProblem: RandomEffectOptimizationProblem[GLM, F],
    val latentFactorOptimizationProblem: GeneralizedLinearOptimizationProblem[GLM, F],
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
  def initializeModel(dimension: Int): GLM = latentFactorOptimizationProblem.initializeZeroModel(dimension)

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
    * Builds a factored random effect optimization problem
    *
    * @param builder
    * @param randomEffectOptimizationConfiguration Random effect configuration
    * @param latentFactorOptimizationConfiguration Latent factor configuration
    * @param mfOptimizationConfiguration MF configuration
    * @param randomEffectDataSet The dataset
    * @return The new optimization problem
    */
  protected[ml] def buildFactoredRandomEffectOptimizationProblem[GLM <: GeneralizedLinearModel,
    F <: DiffFunction[LabeledPoint]](
      builder: (GLMOptimizationConfiguration, Int, Boolean, Boolean) => GeneralizedLinearOptimizationProblem[GLM, F],
      randomEffectOptimizationConfiguration: GLMOptimizationConfiguration,
      latentFactorOptimizationConfiguration: GLMOptimizationConfiguration,
      mfOptimizationConfiguration: MFOptimizationConfiguration,
      randomEffectDataSet: RandomEffectDataSet,
      treeAggregateDepth: Int = 1,
      isTrackingState: Boolean = false,
      isComputingVariance: Boolean = false): FactoredRandomEffectOptimizationProblem[GLM, F] = {

    val MFOptimizationConfiguration(numInnerIterations, latentSpaceDimension) = mfOptimizationConfiguration
    val latentFactorOptimizationProblem = builder(
      latentFactorOptimizationConfiguration,
      treeAggregateDepth,
      isTrackingState,
      isComputingVariance)
    val randomEffectOptimizationProblem = RandomEffectOptimizationProblem.buildRandomEffectOptimizationProblem(
      builder,
      randomEffectOptimizationConfiguration,
      randomEffectDataSet,
      treeAggregateDepth,
      isComputingVariance)

    new FactoredRandomEffectOptimizationProblem(
      randomEffectOptimizationProblem,
      latentFactorOptimizationProblem,
      numInnerIterations,
      latentSpaceDimension)
  }
}
