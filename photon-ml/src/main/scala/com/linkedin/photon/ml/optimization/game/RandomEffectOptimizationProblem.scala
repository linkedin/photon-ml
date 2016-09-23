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
import com.linkedin.photon.ml.optimization.GeneralizedLinearOptimizationProblem
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Representation for a random effect optimization problem
 *
 * - Why sharding the optimizers?
 * Because we may want to preserve the optimization state of each sharded optimization problem
 *
 * - Why sharding the objective functions?
 * Because the regularization weight for each sharded optimization problem may be different, which leads to different
 * objective functions.
 *
 * @param optimizationProblems The component optimization problems (one per individual) for a random effect
 *                             optimization problem
 */
protected[ml] class RandomEffectOptimizationProblem[GLM <: GeneralizedLinearModel, F <: DiffFunction[LabeledPoint]](
    val optimizationProblems: RDD[(String, GeneralizedLinearOptimizationProblem[GLM, F])])
  extends RDDLike {

  def sparkContext: SparkContext = optimizationProblems.sparkContext

  override def setName(name: String): this.type = {
    optimizationProblems.setName(s"$name: Optimization problems")
    this
  }

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!optimizationProblems.getStorageLevel.isValid) {
      optimizationProblems.persist(storageLevel)
    }
    this
  }

  override def unpersistRDD(): this.type = {
    if (optimizationProblems.getStorageLevel.isValid) {
      optimizationProblems.unpersist()
    }
    this
  }

  override def materialize(): this.type = {
    optimizationProblems.count()
    this
  }

  /**
   * Create a default generalized linear model with 0-valued coefficients
   *
   * @param dimension The dimensionality of the model coefficients
   * @return A model with zero coefficients
   */
  def initializeModel(dimension: Int): GLM = optimizationProblems.first()._2.initializeZeroModel(dimension)

  /**
   * Compute the regularization term value
   *
   * @param modelsRDD The trained models
   * @return The combined regularization term value
   */
  def getRegularizationTermValue(modelsRDD: RDD[(String, GeneralizedLinearModel)]): Double = {
    optimizationProblems
      .join(modelsRDD)
      .map {
        case (_, (optimizationProblem, model)) => optimizationProblem.getRegularizationTermValue(model)
      }
      .reduce(_ + _)
  }
}

object RandomEffectOptimizationProblem {
  // Random effect models should not track optimization states per random effect ID. This info is not currently used
  // anywhere and would waste memory.
  //
  // In addition, when enabled the 'run' method in the GeneralizedLinearOptimizationProblem will fail due to an implicit
  // cast of mutable.ListBuffer to mutable.ArrayBuffer, the cause of which is currently undetermined.
  val TRACK_STATE = false

  /**
   * Build an instance of random effect optimization problem
   *
   * @param builder builder of the random effect optimization problem
   * @param configuration Optimizer configuration
   * @param randomEffectDataSet The training dataset
   * @param treeAggregateDepth The depth used in treeAggregate
   * @return A new optimization problem instance
   */
  protected[ml] def buildRandomEffectOptimizationProblem[GLM <: GeneralizedLinearModel,
  F <: DiffFunction[LabeledPoint]](
      builder: (GLMOptimizationConfiguration, Int, Boolean, Boolean) => GeneralizedLinearOptimizationProblem[GLM, F],
      configuration: GLMOptimizationConfiguration,
      randomEffectDataSet: RandomEffectDataSet,
      treeAggregateDepth: Int = 1,
      isComputingVariance: Boolean = false): RandomEffectOptimizationProblem[GLM, F] = {

    // Build an optimization problem for each random effect type
    val optimizationProblems = randomEffectDataSet.activeData.mapValues(_ =>
      builder(configuration, treeAggregateDepth, TRACK_STATE, isComputingVariance)
    )

    new RandomEffectOptimizationProblem(optimizationProblems)
  }
}
