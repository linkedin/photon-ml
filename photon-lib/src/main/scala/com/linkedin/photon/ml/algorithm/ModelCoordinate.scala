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
 * The optimization problem coordinate for a pre-trained model.
 *
 * @tparam D The training data set type
 * @param dataSet The training data set
 */
abstract class ModelCoordinate[D <: DataSet[D]](dataSet: D) extends Coordinate(dataSet) {

  /**
   * Score the effect-specific data set in the coordinate with the input model.
   *
   * @param model The input model
   * @return The output scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores

  /**
   * Initialize a basic model for scoring GAME data.
   *
   * @param seed A random seed
   * @return The basic model
   */
  override protected[algorithm] def initializeModel(seed: Long): DatumScoringModel =
    throw new UnsupportedOperationException("Attempted to initialize model using pre-trained coordinate.")

  /**
   * Update the coordinate with a new dataset.
   *
   * @param dataSet The updated dataset
   * @return A new coordinate with the updated dataset
   */
  override protected[algorithm] def updateCoordinateWithDataSet(dataSet: D): Coordinate[D] =
    throw new UnsupportedOperationException("Attempted to update model coordinate.")

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A tuple of the updated model and the optimization states tracker
   */
  override protected[algorithm] def updateModel(
      model: DatumScoringModel): (DatumScoringModel, Option[OptimizationTracker]) =
    throw new UnsupportedOperationException("Attempted to update model coordinate.")

  /**
   * Compute the regularization term value of the coordinate for a given model.
   *
   * @param model The model
   * @return The regularization term value
   */
  override protected[algorithm] def computeRegularizationTermValue(model: DatumScoringModel): Double = 0D
}
