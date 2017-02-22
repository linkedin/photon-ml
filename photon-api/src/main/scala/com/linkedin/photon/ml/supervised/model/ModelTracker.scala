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
package com.linkedin.photon.ml.supervised.model

import scala.language.existentials

import com.linkedin.photon.ml.optimization.OptimizationStatesTracker

// The 'existentials' package is imported to suppress the following warning message:
//
// photon_trunk/photon-ml/src/main/scala/com/linkedin/photon/ml/supervised/model/ModelTracker.scala:11: inferred
//   existential type Option[(com.linkedin.photon.ml.optimization.OptimizationStatesTracker, Array[_$1])] forSome
//   { type _$1 <: com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel }, which cannot be expressed by
//   wildcards,  should be enabled by making the implicit value scala.language.existentials visible.
// This can be achieved by adding the import clause 'import scala.language.existentials' or by setting the compiler
// option -language:existentials.
// See the Scala docs for value scala.language.existentials for a discussion why the feature should be explicitly
// enabled.

/**
 * A model tracker to include optimization state and per iteration models.
 *
 * @param optimizationStateTracker Tracking object containing state information for each iteration of the optimization
 *                                 process
 * @param models An Array of trained models, one for each iteration in the [[OptimizationStatesTracker]]
 */
case class ModelTracker(optimizationStateTracker: OptimizationStatesTracker, models: Array[_ <: GeneralizedLinearModel])
