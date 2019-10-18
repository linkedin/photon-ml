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
package com.linkedin.photon.ml.data.scoring

import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.GameDatum

/**
 * The class used to track scored data points throughout scoring and validation. The score objects are
 * [[ScoredGameDatum]], full data points with score information.
 *
 * @param scoresRdd Data point scores, as described above
 */
class ModelDataScores(override val scoresRdd: RDD[(UniqueSampleId, ScoredGameDatum)])
  extends DataScores[ScoredGameDatum, ModelDataScores](scoresRdd) {

  /**
   * Generic method to combine two [[ModelDataScores]] objects.
   *
   * @param op The operator to combine two [[ModelDataScores]]
   * @param that The [[ModelDataScores]] instance to merge with this instance
   * @return A merged [[ModelDataScores]]
   */
  private def joinAndApply(
      op: (ScoredGameDatum, ScoredGameDatum) => ScoredGameDatum,
      that: ModelDataScores): ModelDataScores =
    // Use fullOuterJoin: it's possible for some data to not be scored by a model
    new ModelDataScores(
      this
        .scoresRdd
        .fullOuterJoin(that.scoresRdd)
        .mapValues { case (thisScoreOpt, thatScoreOpt) =>
          // Currently acceptable to drop op if one value is missing, since the currently existing operations are
          // commutative and the default value is the 0 value
          (thisScoreOpt, thatScoreOpt) match {
            case (Some(thisScore), Some(thatScore)) => op(thisScore, thatScore)
            case (Some(thisScore), None) => op(thisScore, thisScore.copy(score = MathConst.DEFAULT_SCORE))
            case (None, Some(thatScore)) => op(thatScore.copy(score = MathConst.DEFAULT_SCORE), thatScore)
          }
        })

  /**
   * The addition operation for [[ModelDataScores]].
   *
   * @note This operation performs a full outer join.
   * @param that The [[ModelDataScores]] instance to add to this instance
   * @return A new [[ModelDataScores]] instance encapsulating the accumulated values
   */
  override def +(that: ModelDataScores): ModelDataScores =
    joinAndApply((a, b) => a.copy(score = a.score + b.score), that)

  /**
   * The minus operation for [[ModelDataScores]].
   *
   * @note This operation performs a full outer join.
   * @param that The [[ModelDataScores]] instance to subtract from this instance
   * @return A new [[ModelDataScores]] instance encapsulating the subtracted values
   */
  override def -(that: ModelDataScores): ModelDataScores =
    joinAndApply((a, b) => a.copy(score = a.score - b.score), that)

  /**
   * Method used to define equality on multiple class levels while conforming to equality contract. Defines under
   * what circumstances this class can equal another class.
   *
   * @param other Some other object
   * @return Whether this object can equal the other object
   */
  override def canEqual(other: Any): Boolean = other.isInstanceOf[ModelDataScores]
}

object ModelDataScores {

  /**
   * A factory method to create a [[ModelDataScores]] object from an [[RDD]] of scores.
   *
   * @param scores The scores, consisting of (unique ID, scored datum) pairs.
   * @return A new [[ModelDataScores]] object
   */
  def apply(scores: RDD[(Long, ScoredGameDatum)]): ModelDataScores = new ModelDataScores(scores)

  /**
   * Convert a [[GameDatum]] and a raw score into a score object. For [[CoordinateDataScores]] this is the raw score.
   *
   * @param datum The datum which was scored
   * @param score The raw score for the datum
   * @return The score object
   */
  protected[ml] def toScore(datum: GameDatum, score: Double): ScoredGameDatum = datum.toScoredGameDatum(score)
}
