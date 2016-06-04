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
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.TaskType.TaskType
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
  *                            optimization problem
  */
protected[ml] class RandomEffectOptimizationProblem[F <: TwiceDiffFunction[LabeledPoint]](
    val optimizationProblems: RDD[(String, OptimizationProblem[F])])
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
    * Compute the regularization term value
    *
    * @param coefficientsRDD The trained models
    * @return The combined regularization term value
    */
  def getRegularizationTermValue(coefficientsRDD: RDD[(String, Coefficients)]): Double = {
    optimizationProblems
      .join(coefficientsRDD)
      .map {
        case (_, (optimizationProblem, coefficients)) => optimizationProblem.getRegularizationTermValue(coefficients)
      }
      .reduce(_ + _)
  }
}

object RandomEffectOptimizationProblem {

  /**
    * Build an instance of random effect optimization problem
    *
    * @param taskType The task type (e.g. LinearRegression, LogisticRegression)
    * @param configuration Optimizer configuration
    * @param randomEffectDataSet The training dataset
    * @return A new optimization problem instance
    */
  protected[ml] def buildRandomEffectOptimizationProblem(
      taskType: TaskType,
      configuration: GLMOptimizationConfiguration,
      randomEffectDataSet: RandomEffectDataSet): RandomEffectOptimizationProblem[TwiceDiffFunction[LabeledPoint]] = {

    // Build an optimization problem for each random effect type
    val optimizationProblems = randomEffectDataSet.activeData.mapValues(_ =>
      OptimizationProblem.buildOptimizationProblem(taskType, configuration)
    )

    new RandomEffectOptimizationProblem(optimizationProblems)
  }
}
