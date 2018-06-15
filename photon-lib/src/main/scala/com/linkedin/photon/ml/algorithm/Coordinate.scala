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

import com.linkedin.photon.ml.data.DataSet
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.model.DatumScoringModel
import com.linkedin.photon.ml.optimization.OptimizationTracker

/**
 * The optimization problem coordinate for each effect model.
 *
 * @tparam D The training data set type
 * @param dataSet The training dataset
 */
protected[ml] abstract class Coordinate[D <: DataSet[D]](dataSet: D) {
  /**
   * Score the effect-specific data set in the coordinate with the input model.
   *
   * @param model The input model
   * @return The output scores
   */
  protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores

  /**
   * Initialize a basic model for scoring GAME data.
   *
   * @param seed A random seed
   * @return The basic model
   */
  protected[ml] def initializeModel(seed: Long): DatumScoringModel

  /**
   * Update the coordinate with a new dataset.
   *
   * @param dataSet The updated dataset
   * @return A new coordinate with the updated dataset
   */
  protected[algorithm] def updateCoordinateWithDataSet(dataSet: D): Coordinate[D]

  /**
   * Optimize an existing model for the new scores of the other coordinates.
   *
   * @param model The existing model
   * @param score The combined scores of the other coordinates for each record
   * @return A tuple of the updated model and the optimization states tracker
   */
  protected[algorithm] def updateModel(
      model: DatumScoringModel,
      score: CoordinateDataScores): (DatumScoringModel, Option[OptimizationTracker]) =
    updateCoordinateWithDataSet(dataSet.addScoresToOffsets(score)).updateModel(model)

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A tuple of the updated model and the optimization states tracker
   */
  protected[algorithm] def updateModel(model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker])

  /**
   * Compute the regularization term value of the coordinate for a given model.
   *
   * @param model The model
   * @return The regularization term value
   */
  protected[algorithm] def computeRegularizationTermValue(model: DatumScoringModel): Double
}
