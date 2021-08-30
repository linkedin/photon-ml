/*
 * Copyright 2020 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.hyperparameter.tuner

import com.linkedin.photon.ml.estimators.GameEstimator

object HyperparameterTunerFactory {

  /**
   * Factory for different [[HyperparameterTuner]] objects.
   *
   * @param className Name of [[HyperparameterTuner]] class to instantiate
   * @return The hyper-parameter tuner
   * @throws ClassNotFoundException
   * @throws InstantiationException
   * @throws IllegalAccessException
   */
  def apply(className: String): HyperparameterTuner[GameEstimator.GameResult] =
    Class.forName(className)
      .newInstance
      .asInstanceOf[HyperparameterTuner[GameEstimator.GameResult]]
}
