package com.linkedin.photon.ml.data

import scala.collection.Map

import breeze.linalg.Vector

/**
 * Representation of a single GAME data point
 *
 * @param response the reponse or label
 * @param offset the offset
 * @param weight the importance weight
 * @param featureShardContainer the sharded feature vectors
 * @param randomEffectIdToIndividualIdMap a map from random effect type id to actual individual id
 *   (e.g. "memberId" -> "abc123")
 * @author xazhang
 */
class GameData(
    val response: Double,
    val offset: Double,
    val weight: Double,
    val featureShardContainer: Map[String, Vector[Double]],
    val randomEffectIdToIndividualIdMap: Map[String, String]) {

  /**
   * Build a labeled point with sharded feature container
   *
   * @param featureShardId the feature shard id
   * @return the new labeled point
   */
  def generateLabeledPointWithFeatureShardId(featureShardId: String): LabeledPoint = {
    LabeledPoint(response, featureShardContainer(featureShardId), offset, weight)
  }
}
