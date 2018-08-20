/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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

import com.linkedin.photon.ml.HyperparameterTunerName.{ATLAS, DUMMY, HyperparameterTunerName}

object HyperparameterTunerFactory {

  // Use DUMMY_TUNER for photon-ml, which does an empty operation for hyper-parameter tuning
  val DUMMY_TUNER = "com.linkedin.photon.ml.hyperparameter.tuner.DummyTuner"

  // TODO: Move AtlasTuner into atlas-ml for the auto-tuning system migration:
  // TODO: val ATLAS_TUNER = "com.linkedin.atlas.ml.hyperparameter.tuner.AtlasTuner".
  // TODO: Temporarily stay in photon-ml for test purpose.
  val ATLAS_TUNER = "com.linkedin.photon.ml.hyperparameter.tuner.AtlasTuner"

  /**
   * Factory for different packages of [[HyperparameterTuner]].
   *
   * @param tunerName The name of the auto-tuning package
   * @return The hyper-parameter tuner
   */
  def apply[T](tunerName: HyperparameterTunerName): HyperparameterTuner[T] = {

    val className = tunerName match {
      case DUMMY => DUMMY_TUNER
      case ATLAS => ATLAS_TUNER
      case other => throw new IllegalArgumentException(s"Invalid HyperparameterTuner name: ${other.toString}")
    }

    try {
      Class.forName(className)
        .newInstance
        .asInstanceOf[HyperparameterTuner[T]]
    } catch {
      case ex: Exception =>
        throw new IllegalArgumentException(s"Invalid HyperparameterTuner class: $className", ex)
    }
  }
}
