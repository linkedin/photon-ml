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
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.optimization.SingleNodeOptimizationProblem
import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Representation for a random effect optimization problem.
 *
 * Q: Why shard the optimization problems?
 * A: In the future, we want to be able to have unique regularization weights per optimization problem. In addition, it
 *    may be useful to have access to the optimization state of each individual problem.
 *
 * @tparam Objective The objective function to optimize
 * @param optimizationProblems The component optimization problems (one per individual) for a random effect
 *                             optimization problem
 */
protected[ml] class RandomEffectOptimizationProblem[Objective <: SingleNodeObjectiveFunction](
    val optimizationProblems: RDD[(REId, SingleNodeOptimizationProblem[Objective])],
    glmConstructor: Coefficients => GeneralizedLinearModel,
    val isTrackingState: Boolean)
  extends RDDLike {

  // TODO: Need to refactor 'isTrackingState' out

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = optimizationProblems.sparkContext

  /**
   * Assign a given name to [[optimizationProblems]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   *
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the name of [[optimizationProblems]] assigned
   */
  override def setName(name: String): this.type = {

    optimizationProblems.setName(s"$name: Optimization problems")

    this
  }

  /**
   * Set the storage level of [[optimizationProblems]], and persist their values across the cluster the first time they
   * are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[optimizationProblems]] set
   */
  override def persistRDD(storageLevel: StorageLevel): this.type = {

    if (!optimizationProblems.getStorageLevel.isValid) optimizationProblems.persist(storageLevel)

    this
  }

  /**
   * Mark [[optimizationProblems]] as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @return This object with [[optimizationProblems]] marked non-persistent
   */
  override def unpersistRDD(): this.type = {

    if (optimizationProblems.getStorageLevel.isValid) optimizationProblems.unpersist()

    this
  }

  /**
   * Materialize [[optimizationProblems]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[optimizationProblems]] materialized
   */
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
  def initializeModel(dimension: Int): GeneralizedLinearModel =
    glmConstructor(Coefficients.initializeZeroCoefficients(dimension))

  /**
   * Compute the regularization term value
   *
   * @param modelsRDD The trained models
   * @return The combined regularization term value
   */
  def getRegularizationTermValue(modelsRDD: RDD[(REId, GeneralizedLinearModel)]): Double =
    optimizationProblems
      .join(modelsRDD)
      .map {
        case (_, (optimizationProblem, model)) => optimizationProblem.getRegularizationTermValue(model)
      }
      .reduce(_ + _)
}
