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

import com.linkedin.photon.ml.util.MathUtils.isAlmostZero

/**
 * Representation of a single scored GAME data point. This data structure handles the transition from a training
 * example with features to an example that has been scored. Adding a score field to GameDatum is infeasible because
 * after scoring, the feature shards are no longer needed and keeping them around increases the cost to shuffle the
 * data structure around
 *
 * @param response The response or label
 * @param offset offset
 * @param weight importance weight
 * @param score computed score for this instance
 * @param idTypeToValueMap The id type to value map that holds different types of ids associated with this data
 *                         point. A few examples of the ids types are: (i) ids used to build the random effect model
 *                         such as userId and itemId; (ii) ids used to compute certain metrics like precision@k such
 *                         as documentId or queryId; (iii) ids that are used to uniquely identify each training record.
 */
case class ScoredGameDatum(
    response: Double = 1.0,
    offset: Double = 0.0,
    weight: Double = 1.0,
    score: Double = ScoredGameDatum.ZERO_SCORE,
    idTypeToValueMap: Map[String, String] = Map()) extends Serializable {

  /**
   * Get a copy of the current instance with a score of [[ScoredGameDatum.ZERO_SCORE]]
   *
   * @return Copy of the current scored datum instance with zero score
   */
  def getZeroScoreDatum: ScoredGameDatum = this.copy(score = ScoredGameDatum.ZERO_SCORE)

  /**
   * Check if two scored data are equal (if all of their fields are equal).
   *
   * @param that Some other object
   * @return True if the input is a [[ScoredGameDatum]] with values identical this instance, false otherwise.
   */
  override def equals(that: Any): Boolean =
    that match {
      case other: ScoredGameDatum =>
        ((this.response.isNaN && other.response.isNaN) || isAlmostZero(this.response - other.response)) &&
          isAlmostZero(this.offset - other.offset) &&
          isAlmostZero(this.weight - other.weight) &&
          isAlmostZero(this.score - other.score) &&
          this.idTypeToValueMap.equals(other.idTypeToValueMap)

      case _ => false
    }

  /**
   * Return a text representation of this instance.
   *
   * @return String representation of the scored datum
   */
  override def toString: String =
    s"[response=$response, offset=$offset, weight=$weight, score=$score, idTypeToValueMap=$idTypeToValueMap]"
}

object ScoredGameDatum {
  val ZERO_SCORE = 0.0

  /**
   * A factory method to create [[ScoredGameDatum]] from [[LabeledPoint]] objects.
   *
   * @param labeledPoint A [[LabeledPoint]]
   * @param score The score for the above point
   * @param idTypeToValueMap The ID type to value map
   * @return A new [[ScoredGameDatum]]
   */
  def apply(labeledPoint: LabeledPoint, score: Double, idTypeToValueMap: Map[String, String]): ScoredGameDatum =
    new ScoredGameDatum(labeledPoint.label, labeledPoint.offset, labeledPoint.weight, score, idTypeToValueMap)
}
