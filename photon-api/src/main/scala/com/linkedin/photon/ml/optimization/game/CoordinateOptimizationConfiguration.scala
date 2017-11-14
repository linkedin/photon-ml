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

/**
 * Generic trait for a configuration to define a coordinate.
 */
sealed trait CoordinateOptimizationConfiguration

/**
 * Configuration for GLM coordinate.
 *
 * @param optimizerConfig Optimizer configuration
 * @param regularizationContext Regularization context
 * @param regularizationWeight Regularization weight
 */
protected[ml] abstract class GLMOptimizationConfiguration(
    val optimizerConfig: OptimizerConfig,
    val regularizationContext: RegularizationContext,
    val regularizationWeight: Double)
  extends CoordinateOptimizationConfiguration
  with Serializable {

  require(0 <= regularizationWeight, s"Negative regularization weight: $regularizationWeight")
}

/**
 * Configuration for a fixed effect GLM coordinate
 *
 * @param optimizerConfig Optimizer configuration
 * @param regularizationContext Regularization context
 * @param regularizationWeight Regularization weight
 * @param downSamplingRate Down-sampling rate
 */
case class FixedEffectOptimizationConfiguration(
    override val optimizerConfig: OptimizerConfig,
    override val regularizationContext: RegularizationContext = NoRegularizationContext,
    override val regularizationWeight: Double = 0D,
    downSamplingRate: Double = 1D)
  extends GLMOptimizationConfiguration(optimizerConfig, regularizationContext, regularizationWeight) {

  require(downSamplingRate > 0.0 && downSamplingRate <= 1.0, s"Unexpected downSamplingRate: $downSamplingRate")
}

/**
 * Configuration for a random effect GLM coordinate
 *
 * @param optimizerConfig Optimizer configuration
 * @param regularizationContext Regularization context
 * @param regularizationWeight Regularization weight
 */
case class RandomEffectOptimizationConfiguration(
    override val optimizerConfig: OptimizerConfig,
    override val regularizationContext: RegularizationContext = NoRegularizationContext,
    override val regularizationWeight: Double = 0D)
  extends GLMOptimizationConfiguration(optimizerConfig, regularizationContext, regularizationWeight)

/**
 * Configuration for a factored random effect GLM coordinate
 *
 * @param reOptConfig the random effect optimization configuration
 * @param lfOptConfig the latent factor optimization configuration
 * @param mfOptConfig the matrix factorization optimization configuration
 */
case class FactoredRandomEffectOptimizationConfiguration(
    reOptConfig: RandomEffectOptimizationConfiguration,
    lfOptConfig: RandomEffectOptimizationConfiguration,
    mfOptConfig: MFOptimizationConfiguration)
  extends CoordinateOptimizationConfiguration
