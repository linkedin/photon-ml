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
package com.linkedin.photon.ml.algorithm

import com.linkedin.photon.ml.data.{DataSet, KeyValueScore}
import com.linkedin.photon.ml.model.DatumScoringModel
import com.linkedin.photon.ml.optimization.game.OptimizationTracker

/**
  * The optimization problem coordinate for each effect model
  *
  * @param dataSet the training dataset
  */
protected[ml] abstract class Coordinate[D <: DataSet[D], C <: Coordinate[D, C]](dataSet: D) {

  /**
    * Score the effect-specific data set in the coordinate with the input model
    *
    * @param model The input model
    * @return The output scores
    */
  protected[algorithm] def score(model: DatumScoringModel): KeyValueScore

  /**
    * Initialize the model
    *
    * @param seed A random seed
    */
  protected[algorithm] def initializeModel(seed: Long): DatumScoringModel

  protected[algorithm] def updateModel(model: DatumScoringModel, score: KeyValueScore): (DatumScoringModel, OptimizationTracker) = {
    val dataSetWithUpdatedOffsets = dataSet.addScoresToOffsets(score)
    updateCoordinateWithDataSet(dataSetWithUpdatedOffsets).updateModel(model)
  }

  protected def updateCoordinateWithDataSet(dataSet: D): C

  protected[algorithm] def updateModel(model: DatumScoringModel): (DatumScoringModel, OptimizationTracker)

  protected[algorithm] def computeRegularizationTermValue(model: DatumScoringModel): Double
}
