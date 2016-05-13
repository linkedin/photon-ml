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
package com.linkedin.photon.ml.cli.game.scoring

/**
  * A compact representation of the scored item
  * @param uid A uid that can be used to uniquely characterize the scored item
  * @param predictionScore Score predicted by the model
  */
case class ScoredItem(uid: String, predictionScore: Double) {
  override def equals(that: Any): Boolean = {
    that match {
      case other: ScoredItem => uid == other.uid && predictionScore == other.predictionScore
      case _ => false
    }
  }

  override def hashCode(): Int = {
    uid.hashCode() ^ predictionScore.hashCode()
  }
}
