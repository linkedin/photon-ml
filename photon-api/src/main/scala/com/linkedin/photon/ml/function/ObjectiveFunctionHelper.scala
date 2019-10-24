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
import com.linkedin.photon.ml.function.glm.{GLMLossFunction, LogisticLossFunction, PoissonLossFunction, SquaredLossFunction}
import com.linkedin.photon.ml.function.svm.SmoothedHingeLossFunction
import com.linkedin.photon.ml.optimization.game.CoordinateOptimizationConfiguration
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Helper for [[ObjectiveFunction]] related tasks.
 */
object ObjectiveFunctionHelper {

  type ObjectiveFunctionFactoryFactory = (CoordinateOptimizationConfiguration, Boolean) => (Option[GeneralizedLinearModel], Option[Int]) => ObjectiveFunction
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
      case TaskType.LOGISTIC_REGRESSION => GLMLossFunction.buildFactory(LogisticLossFunction, treeAggregateDepth)
      case TaskType.LINEAR_REGRESSION => GLMLossFunction.buildFactory(SquaredLossFunction, treeAggregateDepth)
      case TaskType.POISSON_REGRESSION => GLMLossFunction.buildFactory(PoissonLossFunction, treeAggregateDepth)
      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => SmoothedHingeLossFunction.buildFactory(treeAggregateDepth)
      case _ => throw new IllegalArgumentException(s"Unknown optimization task type: $taskType")
    }
}
