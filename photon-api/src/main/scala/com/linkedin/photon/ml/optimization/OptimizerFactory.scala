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
package com.linkedin.photon.ml.optimization

import org.apache.spark.broadcast.Broadcast

import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.normalization.NormalizationContext

/**
 * Creates instances of Optimizer according to the user-requested optimizer type and regularization. The factory
 * methods in this class do not enforce the runtime rules for compatibility between user-selected loss functions and
 * optimizers: mixing incompatible optimizers and objective functions will result in a runtime error.
 */
protected[ml] object OptimizerFactory {
  /**
   * Creates an optimizer.
   *
   * @param config The Optimizer configuration
   * @param normalizationContext The normalization context
   * @param regularizationContext The regularization context
   * @param regularizationWeight The regularization weight
   * @param isTrackingState Should the Optimizer track intermediate states during optimization?
   * @return A new Optimizer
   */
  def build(
      config: OptimizerConfig,
      normalizationContext: Broadcast[NormalizationContext],
      regularizationContext: RegularizationContext,
      regularizationWeight: Double = 0,
      isTrackingState: Boolean = Optimizer.DEFAULT_TRACKING_STATE)
    : Optimizer[TwiceDiffFunction] =

    (config.optimizerType, regularizationContext.regularizationType) match {
      case (OptimizerType.LBFGS, RegularizationType.L1 | RegularizationType.ELASTIC_NET) =>
        new OWLQN(
          l1RegWeight = regularizationContext.getL1RegularizationWeight(regularizationWeight),
          normalizationContext = normalizationContext,
          tolerance = config.tolerance,
          maxNumIterations = config.maximumIterations,
          isTrackingState = isTrackingState)

      case (OptimizerType.LBFGS, RegularizationType.L2 | RegularizationType.NONE) =>
        new LBFGS(
          normalizationContext = normalizationContext,
          tolerance = config.tolerance,
          maxNumIterations = config.maximumIterations,
          isTrackingState = isTrackingState)

      case (OptimizerType.TRON, RegularizationType.L2 | RegularizationType.NONE) =>
        new TRON(
          normalizationContext = normalizationContext,
          tolerance = config.tolerance,
          maxNumIterations = config.maximumIterations,
          isTrackingState = isTrackingState)

      case (OptimizerType.TRON, RegularizationType.L1 | RegularizationType.ELASTIC_NET) =>
        throw new IllegalArgumentException("TRON optimizer incompatible with L1 regularization")

      case (OptimizerType.LBFGS | OptimizerType.TRON, regType) =>
        throw new IllegalArgumentException(s"Incompatible regularization selected: $regType")

      case (optType, _) =>
        throw new IllegalArgumentException(s"Incompatible optimizer selected: $optType")
    }
}
