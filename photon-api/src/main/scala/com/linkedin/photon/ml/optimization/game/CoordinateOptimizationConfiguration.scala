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

import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Generic trait for a configuration to define a coordinate.
 */
sealed trait CoordinateOptimizationConfiguration

/**
 * Configuration for a GLM coordinate.
 *
 * @param optimizerConfig Optimizer configuration
 * @param regularizationContext Regularization context
 * @param regularizationWeight Regularization weight
 * @param regularizationWeightRange Regularization weight range
 * @param elasticNetParamRange Elastic net alpha range
 * @param incrementalWeight Optional weight to balance the prior model and the current data in the incremental learning,
 *                          the larger the weight is, the more important the prior model is.
 */
protected[ml] abstract class GLMOptimizationConfiguration(
    val optimizerConfig: OptimizerConfig,
    val regularizationContext: RegularizationContext,
    val regularizationWeight: Double,
    val regularizationWeightRange: Option[DoubleRange] = None,
    val elasticNetParamRange: Option[DoubleRange] = None,
    val incrementalWeight: Option[Double] = None)
  extends CoordinateOptimizationConfiguration
  with Serializable {

  require(0 <= regularizationWeight, s"Negative regularization weight: $regularizationWeight")
  regularizationWeightRange.foreach { case DoubleRange(start, _) =>
    require(start > 0.0, "Regularization weight ranges must be positive")
  }
  elasticNetParamRange.foreach { case DoubleRange(start, end) =>
    require(start >= 0.0 && end <= 1.0, "Elastic net alpha ranges must lie within [0, 1]")
  }
  incrementalWeight.foreach { weight =>
    require(0 <= weight, s"Negative incremental weight: $weight")
  }
}

/**
 * Configuration for a [[com.linkedin.photon.ml.algorithm.FixedEffectCoordinate]].
 *
 * @param optimizerConfig Optimizer configuration
 * @param regularizationContext Regularization context
 * @param regularizationWeight Regularization weight
 * @param regularizationWeightRange Regularization weight range
 * @param elasticNetParamRange Elastic net alpha range
 * @param incrementalWeight Optional weight to balance the prior model and the current data in the incremental learning,
 *                          the larger the weight is, the more important the prior model is.
 * @param downSamplingRate Down-sampling rate
 */
case class FixedEffectOptimizationConfiguration(
    override val optimizerConfig: OptimizerConfig,
    override val regularizationContext: RegularizationContext = NoRegularizationContext,
    override val regularizationWeight: Double = 0D,
    override val regularizationWeightRange: Option[DoubleRange] = None,
    override val elasticNetParamRange: Option[DoubleRange] = None,
    override val incrementalWeight: Option[Double] = None,
    downSamplingRate: Double = 1D)
  extends GLMOptimizationConfiguration(
    optimizerConfig,
    regularizationContext,
    regularizationWeight,
    regularizationWeightRange,
    elasticNetParamRange,
    incrementalWeight) {

  require(downSamplingRate > 0.0 && downSamplingRate <= 1.0, s"Unexpected downSamplingRate: $downSamplingRate")
}

/**
 * Configuration for a [[com.linkedin.photon.ml.algorithm.RandomEffectCoordinate]].
 *
 * @param optimizerConfig Optimizer configuration
 * @param regularizationContext Regularization context
 * @param regularizationWeight Regularization weight
 * @param regularizationWeightRange Regularization weight range
 * @param elasticNetParamRange Elastic net alpha range
 * @param incrementalWeight The weight to balance the prior model and the current data in the incremental learning,
 *                          the larger the weight is, the more important the prior model is. The default value is 1.
 */
case class RandomEffectOptimizationConfiguration(
    override val optimizerConfig: OptimizerConfig,
    override val regularizationContext: RegularizationContext = NoRegularizationContext,
    override val regularizationWeight: Double = 0D,
    override val regularizationWeightRange: Option[DoubleRange] = None,
    override val elasticNetParamRange: Option[DoubleRange] = None,
    override val incrementalWeight: Option[Double] = None)
  extends GLMOptimizationConfiguration(
    optimizerConfig,
    regularizationContext,
    regularizationWeight,
    regularizationWeightRange,
    elasticNetParamRange,
    incrementalWeight)
