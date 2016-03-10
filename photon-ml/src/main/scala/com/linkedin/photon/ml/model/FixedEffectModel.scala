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
package com.linkedin.photon.ml.model

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.BroadcastLike
import com.linkedin.photon.ml.data.{KeyValueScore, GameData}


/**
 * Representation of a fixed effect model
 *
 * @param coefficientsBroadcast the coefficients
 * @param featureShardId the feature shard id
 * @author xazhang
 */
class FixedEffectModel(val coefficientsBroadcast: Broadcast[Coefficients], val featureShardId: String)
  extends Model with BroadcastLike {

  def coefficients: Coefficients = coefficientsBroadcast.value

  /**
   * Compute the score for the dataset
   *
   * @param dataPoints the dataset
   * @return the score
   */
  override def score(dataPoints: RDD[(Long, GameData)]): KeyValueScore = {
    FixedEffectModel.score(dataPoints, coefficientsBroadcast, featureShardId)
  }

  /**
   * Build a summary string for the coefficients
   *
   * @return string representation
   */
  override def toSummaryString: String = {
    s"Fixed effect model with featureShardId $featureShardId summary:\n${coefficients.toSummaryString}"
  }

  /**
   * Clean up coefficient broadcast
   */
  override def unpersistBroadcast(): this.type = {
    coefficientsBroadcast.unpersist()
    this
  }

  /**
   * Create an updated model with the coefficients
   *
   * @param updatedCoefficientsBroadcast new coefficients
   * @return updated model
   */
  def update(updatedCoefficientsBroadcast: Broadcast[Coefficients]): FixedEffectModel = {
    new FixedEffectModel(updatedCoefficientsBroadcast, featureShardId)
  }
}

object FixedEffectModel {

  /**
   * Compute the score for the dataset
   *
   * @param dataPoints the dataset
   * @param coefficientsBroadcast model coefficients
   * @param featureShardId the feature shard id
   * @return the score
   */
  private def score(
      dataPoints: RDD[(Long, GameData)],
      coefficientsBroadcast: Broadcast[Coefficients],
      featureShardId: String): KeyValueScore = {

    val scores = dataPoints.mapValues(gameData =>
      coefficientsBroadcast.value.computeScore(gameData.featureShardContainer(featureShardId))
    )

    new KeyValueScore(scores)
  }
}
