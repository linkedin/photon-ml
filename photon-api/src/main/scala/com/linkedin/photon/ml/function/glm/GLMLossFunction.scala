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
package com.linkedin.photon.ml.function.glm

import com.linkedin.photon.ml.algorithm.Coordinate
import com.linkedin.photon.ml.function.ObjectiveFunction
import com.linkedin.photon.ml.optimization.game.{CoordinateOptimizationConfiguration, FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}

/**
 * Helper for generalized linear model loss function related tasks.
 */
object GLMLossFunction {

  /**
   * Construct a factory function for building distributed and non-distributed generalized linear model loss functions.
   *
   * @param lossFunction A [[PointwiseLossFunction]] for training a generalized linear model
   * @param treeAggregateDepth The tree-aggregate depth to use during aggregation
   * @return A function which builds the appropriate type of [[ObjectiveFunction]] for a given [[Coordinate]] type and
   *         optimization settings.
   */
  def buildFactory
      (lossFunction: PointwiseLossFunction, treeAggregateDepth: Int)
      (config: CoordinateOptimizationConfiguration): Option[Int] => ObjectiveFunction =

    config match {
      case fEOptConfig: FixedEffectOptimizationConfiguration =>
        (interceptIndexOpt: Option[Int]) =>
          DistributedGLMLossFunction(fEOptConfig, lossFunction, treeAggregateDepth, interceptIndexOpt)

      case rEOptConfig: RandomEffectOptimizationConfiguration =>
        (interceptIndexOpt: Option[Int]) =>
          SingleNodeGLMLossFunction(rEOptConfig, lossFunction, interceptIndexOpt)

      case _ =>
        throw new UnsupportedOperationException(
          s"Cannot create a GLM loss function from a coordinate configuration with class '${config.getClass.getName}'")
    }
}
