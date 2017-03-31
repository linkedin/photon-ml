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

import com.linkedin.photon.ml.data.GameDatum

/**
 * The class used to track scored data points throughout training. The score objects are scores only, with no additional
 * information.
 *
 * @param scores The scores consist of (unique ID, score) pairs as explained above.
 */
protected[ml] class CoordinateDataScores(override val scores: RDD[(Long, Double)])
  extends DataScores[Double, CoordinateDataScores](scores) {

  /**
   * Generic method to combine two [[CoordinateDataScores]] objects.
   *
   * @param op The operator to combine two [[CoordinateDataScores]]
   * @param that The [[CoordinateDataScores]] instance to merge with this instance
   * @return A merged [[CoordinateDataScores]]
   */
  private def joinAndApply(op: (Double, Double) => Double, that: CoordinateDataScores): CoordinateDataScores =
    new CoordinateDataScores(
      this
        .scores
        .fullOuterJoin(that.scores)
        .mapValues { case (thisScore, thatScore) => op(thisScore.getOrElse(0.0), thatScore.getOrElse(0.0)) })

  /**
   * The addition operation for [[CoordinateDataScores]].
   *
   * @note This operation performs a full outer join.
   * @param that The [[CoordinateDataScores]] instance to add to this instance
   * @return A new [[CoordinateDataScores]] instance encapsulating the accumulated values
   */
  override def +(that: CoordinateDataScores): CoordinateDataScores = joinAndApply((a, b) => a + b, that)

  /**
   * The minus operation for [[CoordinateDataScores]].
   *
   * @note This operation performs a full outer join.
   * @param that The [[CoordinateDataScores]] instance to subtract from this instance
   * @return A new [[CoordinateDataScores]] instance encapsulating the subtracted values
   */
  override def -(that: CoordinateDataScores): CoordinateDataScores = joinAndApply((a, b) => a - b, that)

  /**
   * Method used to define equality on multiple class levels while conforming to equality contract. Defines under
   * what circumstances this class can equal another class.
   *
   * @param other Some other object
   * @return Whether this object can equal the other object
   */
  override def canEqual(other: Any): Boolean = other.isInstanceOf[CoordinateDataScores]
}

object CoordinateDataScores {

  /**
   * A factory method to create a [[CoordinateDataScores]] object from an [[RDD]] of scores.
   *
   * @param scores The scores, consisting of (unique ID, score) pairs.
   * @return A new [[CoordinateDataScores]] object
   */
  def apply(scores: RDD[(Long, Double)]): CoordinateDataScores = new CoordinateDataScores(scores)

  /**
   * Convert a [[GameDatum]] and a raw score into a score object. For [[CoordinateDataScores]] this is the raw score.
   *
   * @param datum The datum which was scored
   * @param score The raw score for the datum
   * @return The score object
   */
  def toScore(datum: GameDatum, score: Double): Double = score
}
