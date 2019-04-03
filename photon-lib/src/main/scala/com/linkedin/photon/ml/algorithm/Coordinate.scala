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

import com.linkedin.photon.ml.data.Dataset
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.model.DatumScoringModel
import com.linkedin.photon.ml.optimization.OptimizationTracker

/**
 * The optimization problem coordinate for each effect model.
 *
 * @tparam D The training dataset type
 * @param dataset The training dataset
 */
protected[ml] abstract class Coordinate[D <: Dataset[D]](dataset: D) {

  /**
   * Update the coordinate with a new dataset.
   *
   * @param dataset The updated dataset
   * @return A new coordinate with the updated dataset
   */
  protected[algorithm] def updateCoordinateWithDataset(dataset: D): Coordinate[D]

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A tuple of the updated model and the optimization states tracker
   */
  protected[algorithm] def trainModel(): (DatumScoringModel, Option[OptimizationTracker])

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @param score The combined scores for each record of the other coordinates
   * @return A tuple of the updated model and the optimization states tracker
   */
  protected[algorithm] def trainModel(score: CoordinateDataScores): (DatumScoringModel, Option[OptimizationTracker]) =
    updateCoordinateWithDataset(dataset.addScoresToOffsets(score)).trainModel()

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A tuple of the updated model and the optimization states tracker
   */
  protected[algorithm] def trainModel(model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker])

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point and with residuals from other coordinates.
   *
   * @param model The existing model
   * @param score The combined scores for each record of the other coordinates
   * @return A tuple of the updated model and the optimization states tracker
   */
  protected[algorithm] def trainModel(
      model: DatumScoringModel,
      score: CoordinateDataScores): (DatumScoringModel, Option[OptimizationTracker]) =
    updateCoordinateWithDataset(dataset.addScoresToOffsets(score)).trainModel(model)

  /**
   * Score the effect-specific dataset in the coordinate with the input model.
   *
   * @param model The input model
   * @return The output scores
   */
  protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores
}
