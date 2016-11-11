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

import com.linkedin.photon.ml.BroadcastLike
import com.linkedin.photon.ml.data.{GameDatum, KeyValueScore}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

/**
 * Representation of a fixed effect model
 *
 * @param modelBroadcast The coefficients
 * @param featureShardId The feature shard id
 */
protected[ml] class FixedEffectModel(val modelBroadcast: Broadcast[GeneralizedLinearModel], val featureShardId: String)
  extends DatumScoringModel with BroadcastLike {

  def model: GeneralizedLinearModel = modelBroadcast.value

  /**
   * Clean up coefficient broadcast
   */
  override def unpersistBroadcast(): this.type = {
    modelBroadcast.unpersist()
    this
  }

  /**
   * Compute the score for the dataset
   *
   * @param dataPoints The dataset
   * @return The score
   */
  override def score(dataPoints: RDD[(Long, GameDatum)]): KeyValueScore =
    FixedEffectModel.score(dataPoints, modelBroadcast, featureShardId)

  /**
   * Build a summary string for the coefficients
   *
   * @return String representation
   */
  override def toSummaryString: String =
    s"Fixed effect model with featureShardId $featureShardId summary:\n${model.toSummaryString}"

  /**
   * Create an updated model with the coefficients
   *
   * @param updatedModelBroadcast new coefficients
   * @return updated model
   */
  def update(updatedModelBroadcast: Broadcast[GeneralizedLinearModel]): FixedEffectModel =
    new FixedEffectModel(updatedModelBroadcast, featureShardId)

  override def equals(that: Any): Boolean = {
    that match {
      case other: FixedEffectModel =>
        val sameMetaData = this.featureShardId == other.featureShardId
        lazy val sameCoefficients = this.model.equals(other.model)
        sameMetaData && sameCoefficients
      case _ => false
    }
  }

  // TODO: Violation of the hashCode() contract
  override def hashCode(): Int = super.hashCode()
}

object FixedEffectModel {

  /**
   * Compute the scores for the dataset
   *
   * TODO(fastier): do we really need a static? (also in RandomEffectModel)
   *
   * @param dataPoints The dataset to score
   * @param modelBroadcast The model to use for scoring
   * @param featureShardId The feature shard id
   * @return The score
   */
  private def score(
      dataPoints: RDD[(Long, GameDatum)],
      modelBroadcast: Broadcast[GeneralizedLinearModel],
      featureShardId: String): KeyValueScore = {

    val scores = dataPoints.mapValues(gameData =>
      modelBroadcast.value.computeScore(gameData.featureShardContainer(featureShardId))
    )

    new KeyValueScore(scores)
  }
}
