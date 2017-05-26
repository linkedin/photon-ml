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
package com.linkedin.photon.ml.data

import scala.collection.Map

import breeze.linalg.Vector

import com.linkedin.photon.ml.data.scoring.ScoredGameDatum

/**
 * Representation of a single GAME data point.
 *
 * @param response The response or label
 * @param offsetOpt An optional field for offset
 * @param weightOpt An optional field for importance weight
 * @param featureShardContainer The sharded feature vectors
 * @param idTagToValueMap A map of ID tag to ID for this data point. An ID tag is a column or metadata field containing
 *                        IDs used to group or uniquely identify samples. Examples of ID tags that may be stored as keys
 *                        in this map are:
 *
 *                        (i) ID tags used to build random effect models (e.g. userId, jobId);
 *                        (ii) ID tags used to compute evaluation metrics like precision@k (e.g. documentId, queryId);
 *                        (iii) ID tags used to uniquely identify training records (e.g. uid)
 */
protected[ml] class GameDatum(
    val response: Double,
    val offsetOpt: Option[Double],
    val weightOpt: Option[Double],
    val featureShardContainer: Map[String, Vector[Double]],
    val idTagToValueMap: Map[String, String]) extends Serializable {

  import GameDatum._

  val offset: Double = offsetOpt.getOrElse(DEFAULT_OFFSET)
  val weight: Double = weightOpt.getOrElse(DEFAULT_WEIGHT)

  /**
   * Build a labeled point with sharded feature container.
   *
   * @param featureShardId The feature shard id
   * @return The new labeled point
   */
  def generateLabeledPointWithFeatureShardId(featureShardId: String): LabeledPoint = {
    LabeledPoint(label = response, features = featureShardContainer(featureShardId), offset = offset, weight = weight)
  }

  /**
   * Generate a scored data point, using this data point as a base.
   *
   * @param score The score for this data point
   * @return A new [[ScoredGameDatum]] instance
   */
  def toScoredGameDatum(score: Double = ScoredGameDatum.ZERO_SCORE): ScoredGameDatum = {
    ScoredGameDatum(response, offset, weight, score, idTagToValueMap)
  }
}

object GameDatum {
  val DEFAULT_OFFSET = 0.0
  val DEFAULT_WEIGHT = 1.0
}
