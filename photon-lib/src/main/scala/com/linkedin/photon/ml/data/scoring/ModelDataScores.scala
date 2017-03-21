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

/**
 * The class used to track scored data points throughout scoring and validation. The score objects are
 * [[ScoredGameDatum]], full data points with score information.
 *
 * @param scores Data point scores, as described above
 */
protected[ml] class ModelDataScores(val scores: RDD[(Long, ScoredGameDatum)])
  extends DataScores[ScoredGameDatum, ModelDataScores](scores) {

  /**
   * Generic method to combine two [[ModelDataScores]] objects.
   *
   * @param op The operator to combine two [[ModelDataScores]]
   * @param that The [[ModelDataScores]] instance to merge with this instance
   * @return A merged [[ModelDataScores]]
   */
  private def fullOuterJoin(
      op: (ScoredGameDatum, ScoredGameDatum) => ScoredGameDatum,
      that: ModelDataScores): ModelDataScores =
    new ModelDataScores(
      this
        .scores
        .cogroup(that.scores)
        .mapValues {
          case (Seq(sd1), Seq(sd2)) => op(sd1, sd2)
          case (Seq(), Seq(sd2)) => op(sd2.getZeroScoreDatum, sd2)
          case (Seq(sd1), Seq()) => op(sd1, sd1.getZeroScoreDatum)
        })

  /**
   * The addition operation for [[ModelDataScores]].
   *
   * @note This operation performs a full outer join.
   * @param that The [[ModelDataScores]] instance to add to this instance
   * @return A new [[ModelDataScores]] instance encapsulating the accumulated values
   */
  override def +(that: ModelDataScores): ModelDataScores =
    fullOuterJoin((a, b) => a.copy(score = a.score + b.score), that)

  /**
   * The minus operation for [[ModelDataScores]].
   *
   * @note This operation performs a full outer join.
   * @param that The [[ModelDataScores]] instance to subtract from this instance
   * @return A new [[ModelDataScores]] instance encapsulating the subtracted values
   */
  override def -(that: ModelDataScores): ModelDataScores =
    fullOuterJoin((a, b) => a.copy(score = a.score - b.score), that)

  /**
   * Compare two [[ModelDataScores]]s objects.
   *
   * @param that Some other object
   * @return True if the both [[ModelDataScores]] objects have identical [[ScoredGameDatum]] for each unique ID, false
   *         otherwise
   */
  override def equals(that: Any): Boolean = {
    that match {
      case other: ModelDataScores =>
        this.scores
          .fullOuterJoin(other.scores)
          .mapPartitions(iterator =>
            Iterator.single(iterator.forall { case (_, (thisScoreOpt1, thisScoreOpt2)) =>
              thisScoreOpt1.isDefined && thisScoreOpt2.isDefined && thisScoreOpt1.get.equals(thisScoreOpt2.get)
            })
          )
          .filter(!_).count() == 0
      case _ => false
    }
  }
}
