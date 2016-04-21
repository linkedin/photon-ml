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

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.data.{RandomEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.projector.ProjectionMatrix
import com.linkedin.photon.ml.supervised.TaskType._

/**
 * An optimization problem for factored random effect datasets
 *
 * @param randomEffectOptimizationProblem the random effect optimization problem
 * @param latentFactorOptimizationProblem the latent factor optimization problem
 * @param numIterations number of iterations
 * @param latentSpaceDimension dimensionality of latent space
 * @author xazhang
 */
protected[ml] class FactoredRandomEffectOptimizationProblem[F <: TwiceDiffFunction[LabeledPoint]](
    val randomEffectOptimizationProblem: RandomEffectOptimizationProblem[F],
    val latentFactorOptimizationProblem: OptimizationProblem[F],
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
   * Compute the regularization term value
   *
   * @param coefficientsRDD the coefficients
   * @param projectionMatrix the projection matrix
   * @return regularization term value
   */
  def getRegularizationTermValue(
      coefficientsRDD: RDD[(String, Coefficients)],
      projectionMatrix: ProjectionMatrix): Double = {

    val projectionMatrixAsCoefficients = new Coefficients(projectionMatrix.matrix.flatten(), variancesOption = None)
    randomEffectOptimizationProblem.getRegularizationTermValue(coefficientsRDD) +
        latentFactorOptimizationProblem.getRegularizationTermValue(projectionMatrixAsCoefficients)
  }
}

object FactoredRandomEffectOptimizationProblem {

  /**
   * Builds a factored random effect optimization problem
   *
   * @param taskType the task type
   * @param randomEffectOptimizationConfiguration random effect configuration
   * @param latentFactorOptimizationConfiguration latent factor configuration
   * @param mfOptimizationConfiguration MF configuration
   * @param randomEffectDataSet the dataset
   * @return the new optimization problem
   */
  protected[ml] def buildFactoredRandomEffectOptimizationProblem(
      taskType: TaskType,
      randomEffectOptimizationConfiguration: GLMOptimizationConfiguration,
      latentFactorOptimizationConfiguration: GLMOptimizationConfiguration,
      mfOptimizationConfiguration: MFOptimizationConfiguration,
      randomEffectDataSet: RandomEffectDataSet)
  : FactoredRandomEffectOptimizationProblem[TwiceDiffFunction[LabeledPoint]] = {

    val randomEffectOptimizationProblem = RandomEffectOptimizationProblem.buildRandomEffectOptimizationProblem(taskType,
      randomEffectOptimizationConfiguration, randomEffectDataSet)
    val latentFactorOptimizationProblem = OptimizationProblem.buildOptimizationProblem(taskType,
      latentFactorOptimizationConfiguration)
    val MFOptimizationConfiguration(numInnerIterations, latentSpaceDimension) = mfOptimizationConfiguration
    new FactoredRandomEffectOptimizationProblem(randomEffectOptimizationProblem, latentFactorOptimizationProblem,
      numInnerIterations, latentSpaceDimension)
  }
}
