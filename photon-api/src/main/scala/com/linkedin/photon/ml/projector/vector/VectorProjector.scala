/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.projector.vector

import breeze.linalg.Vector

import com.linkedin.photon.ml.projector.Projector

/**
 * Trait for an object which performs two types of projections:
 *   1. Project a [[Vector]] from the original space to the projected space (e.g. project a feature vector to the
 *      projected space prior to the model training phase).
 *   2. Project a [[Vector]] from the projected space to the original space (e.g. project a coefficient vector to the
 *      original space after the model training phase).
 */
trait VectorProjector extends Projector[Vector[Double]]
