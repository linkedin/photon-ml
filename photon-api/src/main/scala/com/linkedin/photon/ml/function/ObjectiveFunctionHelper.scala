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
package com.linkedin.photon.ml.function

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.algorithm.Coordinate
import com.linkedin.photon.ml.function.glm.{LogisticLossFunction, PointwiseLossFunction, PoissonLossFunction, SquaredLossFunction}
import com.linkedin.photon.ml.optimization.game.{CoordinateOptimizationConfiguration, FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Helper for [[ObjectiveFunction]] related tasks.
 */
object ObjectiveFunctionHelper {

  type ObjectiveFunctionFactoryFactory = CoordinateOptimizationConfiguration => (Option[GeneralizedLinearModel], Option[Int]) => ObjectiveFunction
  type DistributedObjectiveFunctionFactory = (Option[GeneralizedLinearModel], Option[Int]) => DistributedObjectiveFunction
  type SingleNodeObjectiveFunctionFactory = (Option[GeneralizedLinearModel], Option[Int]) => SingleNodeObjectiveFunction

  /**
   * Construct a factory function for building [[ObjectiveFunction]] objects.
   *
   * @param taskType The training task to perform
   * @param treeAggregateDepth The tree-aggregate depth to use during aggregation
   * @return A function which builds the appropriate type of [[ObjectiveFunction]] for a given [[Coordinate]] type and
   *         optimization settings.
   */
  def buildFactory(taskType: TaskType, treeAggregateDepth: Int): ObjectiveFunctionFactoryFactory =
    taskType match {
      case TaskType.LOGISTIC_REGRESSION => factoryHelper(LogisticLossFunction, treeAggregateDepth)
      case TaskType.LINEAR_REGRESSION => factoryHelper(SquaredLossFunction, treeAggregateDepth)
      case TaskType.POISSON_REGRESSION => factoryHelper(PoissonLossFunction, treeAggregateDepth)
      case _ => throw new IllegalArgumentException(s"Unknown optimization task type: $taskType")
    }

  /**
   * Construct a factory function for building distributed and non-distributed generalized linear model loss functions.
   *
   * @param lossFunction A [[PointwiseLossFunction]] for training a generalized linear model
   * @param treeAggregateDepth The tree-aggregate depth to use during aggregation
   * @return A function which builds the appropriate type of [[ObjectiveFunction]] for a given [[Coordinate]] type and
   *         optimization settings.
   */
  private def factoryHelper
      (lossFunction: PointwiseLossFunction, treeAggregateDepth: Int)
      (config: CoordinateOptimizationConfiguration): (Option[GeneralizedLinearModel], Option[Int]) => ObjectiveFunction =
    config match {
      case fEOptConfig: FixedEffectOptimizationConfiguration =>
        (priorModelOpt: Option[GeneralizedLinearModel], interceptIndexOpt: Option[Int]) =>
          DistributedObjectiveFunction(fEOptConfig, lossFunction, treeAggregateDepth, priorModelOpt,  interceptIndexOpt)

      case rEOptConfig: RandomEffectOptimizationConfiguration =>
        (priorModelOpt: Option[GeneralizedLinearModel], interceptIndexOpt: Option[Int]) =>
          SingleNodeObjectiveFunction(rEOptConfig, lossFunction, priorModelOpt, interceptIndexOpt)

      case _ =>
        throw new UnsupportedOperationException(
          s"Cannot create a GLM loss function from a coordinate configuration with class '${config.getClass.getName}'")
    }
}
