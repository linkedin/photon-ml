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
package com.linkedin.photon.ml.hyperparameter

import com.linkedin.photon.ml.HyperparameterTuningMode.HyperparameterTuningMode
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Configurations of hyper-parameters.
 *
 * @param tuningMode Hyper-parameter auto-tuning mode.
 * @param names A Seq of hyper-parameter names.
 * @param ranges A Seq of searching ranges for hyper-parameters.
 * @param discreteParams A Map that specifies the indices of discrete parameters and their numbers of discrete values.
 * @param transformMap A Map that specifies the indices of parameters and their names of transform functions.
 */
case class HyperparameterConfig(
    tuningMode: HyperparameterTuningMode,
    names: Seq[String],
    ranges: Seq[DoubleRange],
    discreteParams: Map[Int, Int],
    transformMap: Map[Int, String])

