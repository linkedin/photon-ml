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
package com.linkedin.photon.ml.algorithm

import com.linkedin.photon.ml.model.DatumScoringModel
import com.linkedin.photon.ml.optimization.OptimizationTracker

/**
 * The optimization problem coordinate for each effect model.
 *
 */
protected[ml] abstract class Coordinate {

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A (updated model, optimization state tracking information) tuple
   */
  protected[algorithm] def trainModel(): (DatumScoringModel, OptimizationTracker)

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A (updated model, optimization state tracking information) tuple
   */
  protected[algorithm] def trainModel(model: DatumScoringModel): (DatumScoringModel, OptimizationTracker)

  /**
   * Generate a new dataset with updated offset.
   *
   * @param model The model of previous coordinate
   * @return A new dataset with the updated offsets
   */
  protected[algorithm] def updateOffset(model: DatumScoringModel)
}

