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

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.data.{RandomEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.TaskType.TaskType


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
 * @param optimizationProblems the component optimization problems for each random effect type
 * @author xazhang
 */
class RandomEffectOptimizationProblem[F <: TwiceDiffFunction[LabeledPoint]](
    val optimizationProblems: RDD[(String, OptimizationProblem[F])])
  extends RDDLike {

  def sparkContext = optimizationProblems.sparkContext

  def setName(name: String): this.type = {
    optimizationProblems.setName(s"$name: Optimization problems")
    this
  }

  def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!optimizationProblems.getStorageLevel.isValid) {
      optimizationProblems.persist(storageLevel)
    }
    this
  }

  def unpersistRDD(): this.type = {
    if (optimizationProblems.getStorageLevel.isValid) {
      optimizationProblems.unpersist()
    }
    this
  }

  def materialize(): this.type = {
    optimizationProblems.count()
    this
  }

  /**
   * Compute the regularization term value
   *
   * @param model the model
   * @return regularization term value
   */
  def getRegularizationTermValue(coefficientsRDD: RDD[(String, Coefficients)]): Double = {
    optimizationProblems
      .join(coefficientsRDD)
      .map {
        case (_, (optimizationProblem, coefficients)) =>
          optimizationProblem.getRegularizationTermValue(coefficients)
      }
      .reduce(_ + _)
  }
}

object RandomEffectOptimizationProblem {

  /**
   * Build an instance of random effect optimization problem
   *
   * @param taskType the task type (e.g. LinearRegression, LogisticRegression)
   * @param configuration optimizer configuration
   * @param randomEffectDataSet the training dataset
   * @return a new optimization problem instance
   */
  def buildRandomEffectOptimizationProblem(
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
