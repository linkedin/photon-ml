package com.linkedin.photon.ml.model

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.BroadcastLike
import com.linkedin.photon.ml.data.{KeyValueScore, GameData}


/**
 * @author xazhang
 */
class FixedEffectModel(val coefficientsBroadcast: Broadcast[Coefficients], val featureShardId: String)
    extends Model with BroadcastLike {

  def coefficients: Coefficients = coefficientsBroadcast.value

  override def score(dataPoints: RDD[(Long, GameData)]): KeyValueScore = {
    FixedEffectModel.score(dataPoints, coefficientsBroadcast, featureShardId)
  }

  override def toSummaryString: String = {
    s"Fixed effect model with featureShardId $featureShardId summary:\n${coefficients.toSummaryString}"
  }

  override def unpersistBroadcast(): this.type = {
    coefficientsBroadcast.unpersist()
    this
  }

  def update(updatedCoefficientsBroadcast: Broadcast[Coefficients]): FixedEffectModel = {
    new FixedEffectModel(updatedCoefficientsBroadcast, featureShardId)
  }
}

object FixedEffectModel {
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
