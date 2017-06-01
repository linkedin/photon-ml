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
package com.linkedin.photon.ml.model

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.data.scoring.{CoordinateDataScores, ModelDataScores}
import com.linkedin.photon.ml.util.Summarizable

/**
 * Models that need to be available for scoring must extend this trait.
 */
trait DatumScoringModel extends Summarizable {

  /**
   * Even though a model may have many sub-problems, there is only one loss function type for a given DatumScoringModel.
   */
  def modelType: TaskType

  /**
   * Compute the scores for the GAME data set.
   *
   * @note "score" = features * coefficients (Before link function in the case of logistic regression, for example)
   * @param dataPoints The dataset to score (Note that the Long in the RDD is a unique identifier for the paired
   *                   [[GameDatum]] object, referred to in the GAME code as the "unique id")
   * @return The computed scores
   */
  def score(dataPoints: RDD[(Long, GameDatum)]): ModelDataScores

  /**
   * Compute the scores for the GAME data set, and store the scores only.
   *
   * @note "score" = features * coefficients (Before link function in the case of logistic regression, for example)
   * @param dataPoints The dataset to score (Note that the Long in the RDD is a unique identifier for the paired
   *                   [[GameDatum]] object, referred to in the GAME code as the "unique id")
   * @return The computed scores
   */
  protected[ml] def scoreForCoordinateDescent(dataPoints: RDD[(Long, GameDatum)]): CoordinateDataScores
}
