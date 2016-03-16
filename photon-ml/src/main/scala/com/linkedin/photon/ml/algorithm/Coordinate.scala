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

import com.linkedin.photon.ml.data.{KeyValueScore, DataSet}
import com.linkedin.photon.ml.model.Model
import com.linkedin.photon.ml.optimization.game.OptimizationTracker


/**
 * The optimization problem coordinate for each effect model
 *
 * @param dataSet the training dataset
 * @author xazhang
 */
abstract class Coordinate[D <: DataSet[D], C <: Coordinate[D, C]](dataSet: D) {

  /**
   * Score the effect-specific data set in the coordinate with the input model
   *
   * @param model the input model
   * @return the output scores
   */
  def score(model: Model): KeyValueScore

  /**
   * Initialize the model
   *
   * @param seed random seed
   */
  def initializeModel(seed: Long): Model

  def updateModel(model: Model, score: KeyValueScore): (Model, OptimizationTracker) = {
    val dataSetWithUpdatedOffsets = dataSet.addScoresToOffsets(score)
    updateCoordinateWithDataSet(dataSetWithUpdatedOffsets).updateModel(model)
  }

  protected def updateCoordinateWithDataSet(dataSet: D): C

  def updateModel(model: Model): (Model, OptimizationTracker)

  def computeRegularizationTermValue(model: Model): Double
}
