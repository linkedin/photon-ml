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

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.data.scoring.{CoordinateDataScores, ModelDataScores}
import com.linkedin.photon.ml.spark.BroadcastLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Representation of a fixed effect model.
 *
 * @param modelBroadcast The coefficients
 * @param featureShardId The feature shard id
 */
class FixedEffectModel(
    val modelBroadcast: Broadcast[GeneralizedLinearModel],
    val featureShardId: String)
  extends DatumScoringModel
  with BroadcastLike {

  override val modelType = modelBroadcast.value.modelType

  /**
   * Get the underlying [[GeneralizedLinearModel]].
   *
   * @return The broadcast [[GeneralizedLinearModel]]
   */
  def model: GeneralizedLinearModel = modelBroadcast.value

  /**
   * Create an updated model with the coefficients.
   *
   * @param updatedModelBroadcast New coefficients
   * @return Updated model
   */
  protected[ml] def update(updatedModelBroadcast: Broadcast[GeneralizedLinearModel]): FixedEffectModel =
    new FixedEffectModel(updatedModelBroadcast, featureShardId)

  /**
   * Compute the scores for the dataset.
   *
   * @note Use a static method to avoid serializing entire model object during RDD operations.
   * @param dataPoints The dataset to score (Note that the Long in the RDD is a unique identifier for the paired
   *                   [[GameDatum]] object, referred to in the GAME code as the "unique id")
   * @return The computed scores
   */
  override def score(dataPoints: RDD[(Long, GameDatum)]): ModelDataScores =
    FixedEffectModel.score(dataPoints, modelBroadcast, featureShardId, ModelDataScores.toScore, ModelDataScores.apply)

  /**
   * Compute the scores for the GAME data set, and store the scores only.
   *
   * @note Use a static method to avoid serializing entire model object during RDD operations.
   * @param dataPoints The dataset to score (Note that the Long in the RDD is a unique identifier for the paired
   *                   [[GameDatum]] object, referred to in the GAME code as the "unique id")
   * @return The computed scores
   */
  override protected[ml] def scoreForCoordinateDescent(dataPoints: RDD[(Long, GameDatum)]): CoordinateDataScores =
    FixedEffectModel.score(
      dataPoints,
      modelBroadcast,
      featureShardId,
      CoordinateDataScores.toScore,
      CoordinateDataScores.apply)

  /**
   * Build a summary string for the coefficients.
   *
   * @return String representation
   */
  override def toSummaryString: String =
    s"Fixed effect model with featureShardId $featureShardId summary:\n${model.toSummaryString}"

  /**
   * Clean up coefficient broadcast.
   */
  override protected[ml] def unpersistBroadcast(): BroadcastLike = {
    modelBroadcast.unpersist()
    this
  }

  /**
   * Compares two [[FixedEffectModel]] objects.
   *
   * @param that Some other object
   * @return True if both models have the same feature shard ID and underlying models, false otherwise
   */
  override def equals(that: Any): Boolean = {
    that match {
      case other: FixedEffectModel =>
        val sameMetaData = this.featureShardId == other.featureShardId
        lazy val sameModel = this.model.equals(other.model)
        sameMetaData && sameModel
      case _ => false
    }
  }

  /**
   * Returns a hash code value for the object.
   *
   * @return An [[Int]] hash code
   */
  override def hashCode: Int = featureShardId.hashCode + model.hashCode
}

object FixedEffectModel {

  /**
   * Compute the scores for the dataset.
   *
   * @param dataPoints The dataset to score
   * @param modelBroadcast The model to use for scoring
   * @param featureShardId The feature shard id
   * @return The scores
   */
  private def score[T, V](
      dataPoints: RDD[(Long, GameDatum)],
      modelBroadcast: Broadcast[GeneralizedLinearModel],
      featureShardId: String,
      toScore: (GameDatum, Double) => T,
      toResult: (RDD[(Long, T)]) => V): V = {

    val scores = dataPoints.mapValues { gameDatum =>
      toScore(gameDatum, modelBroadcast.value.computeScore(gameDatum.featureShardContainer(featureShardId)))
    }

    toResult(scores)
  }
}
