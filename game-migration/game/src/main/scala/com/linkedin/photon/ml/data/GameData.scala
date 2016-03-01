package com.linkedin.photon.ml.data


import scala.collection.Map

import breeze.linalg.Vector


/**
 * @author xazhang
 */
class GameData(
    val response: Double,
    val offset: Double,
    val weight: Double,
    val featureShardContainer: Map[String, Vector[Double]],
    val randomEffectIdToIndividualIdMap: Map[String, String]) {

  def generateLabeledPointWithFeatureShardId(featureShardId: String): LabeledPoint = {
    LabeledPoint(response, featureShardContainer(featureShardId), offset, weight)
  }
}
