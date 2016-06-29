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

import breeze.linalg.Vector

import scala.collection.Map

/**
  * Representation of a single GAME data point
  *
  * @param response The response or label
  * @param offset The offset
  * @param weight The importance weight
  * @param featureShardContainer The sharded feature vectors
  * @param randomEffectIdToIndividualIdMap A map from random effect type id to actual individual id
  *   (e.g. "memberId" -> "1234" or "itemId" -> "abcd")
  */
protected[ml] class GameDatum(
    val response: Double,
    val offset: Double,
    val weight: Double,
    val featureShardContainer: Map[String, Vector[Double]],
    val randomEffectIdToIndividualIdMap: Map[String, String]) extends Serializable {

  /**
    * Build a labeled point with sharded feature container
    *
    * @param featureShardId The feature shard id
    * @return The new labeled point
    */
  def generateLabeledPointWithFeatureShardId(featureShardId: String): LabeledPoint = {
    LabeledPoint(response, featureShardContainer(featureShardId), offset, weight)
  }
}
