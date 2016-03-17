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
 *   (e.g. "memberId" -> "1234" or "itemId" -> "abcd")
 * @author xazhang
 */
protected[ml] class GameData(
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
