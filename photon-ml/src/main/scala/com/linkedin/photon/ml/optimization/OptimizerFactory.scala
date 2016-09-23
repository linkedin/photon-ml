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
package com.linkedin.photon.ml.optimization

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.{DiffFunction, TwiceDiffFunction}

/**
 * Creates instances of Optimizer according to the objective function type and configuration. The factory methods in
 * this class enforce the runtime rules for compatibility between user-selected loss functions and optimizers.
 */
protected[ml] object OptimizerFactory {

  /**
   * Creates an optimizer for DiffFunction objective functions.
   *
   * @param config Optimizer configuration
   * @return A new DiffFunction Optimizer created according to the configuration
   */
  def diffOptimizer(config: OptimizerConfig): Optimizer[LabeledPoint, DiffFunction[LabeledPoint]] = {
    val optimizer = config.optimizerType match {
      case OptimizerType.LBFGS =>
        new LBFGS[LabeledPoint]

      case optType =>
        throw new IllegalArgumentException(s"Selected optimizer $optType incompatible with DiffFuntion.");
    }

    optimizer.setMaximumIterations(config.maximumIterations)
    optimizer.setTolerance(config.tolerance)
    optimizer.setConstraintMap(config.constraintMap)

    optimizer
  }

  /**
   * Creates an optimizer for TwiceDiffFunction objective functions.
   *
   * @param config Optimizer configuration
   * @return A new TwiceDiffFunction Optimizer created according to the configuration
   */
  def twiceDiffOptimizer(config: OptimizerConfig): Optimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]] = {
    val optimizer = config.optimizerType match {
      case OptimizerType.LBFGS =>
        new LBFGS[LabeledPoint]

      case OptimizerType.TRON =>
        new TRON[LabeledPoint]

      case optType =>
        throw new IllegalArgumentException(s"Selected optimizer $optType incompatible with TwiceDiffFuntion.");
    }

    optimizer.setMaximumIterations(config.maximumIterations)
    optimizer.setTolerance(config.tolerance)
    optimizer.setConstraintMap(config.constraintMap)

    optimizer
  }
}
