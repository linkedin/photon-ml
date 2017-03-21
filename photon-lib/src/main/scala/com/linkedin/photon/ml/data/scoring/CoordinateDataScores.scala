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
 * The class used to track scored data points throughout training. The score objects are scores only, with no additional
 * information.
 *
 * @param scores The scores consist of (unique ID, score) pairs as explained above.
 */
protected[ml] class CoordinateDataScores(val scores: RDD[(Long, Double)])
  extends DataScores[Double, CoordinateDataScores](scores) {

  /**
   * Generic method to combine two [[CoordinateDataScores]] objects.
   *
   * @param op The operator to combine two [[CoordinateDataScores]]
   * @param that The [[CoordinateDataScores]] instance to merge with this instance
   * @return A merged [[CoordinateDataScores]]
   */
  private def fullOuterJoin(op: (Double, Double) => Double, that: CoordinateDataScores): CoordinateDataScores =
    new CoordinateDataScores(
      this
        .scores
        .cogroup(that.scores)
        .mapValues {
          case (Seq(sd1), Seq(sd2)) => op(sd1, sd2)
          case (Seq(), Seq(sd2)) => op(0.0, sd2)
          case (Seq(sd1), Seq()) => op(sd1, 0.0)
        })

  /**
   * The addition operation for [[CoordinateDataScores]].
   *
   * @note This operation performs a full outer join.
   * @param that The [[CoordinateDataScores]] instance to add to this instance
   * @return A new [[CoordinateDataScores]] instance encapsulating the accumulated values
   */
  override def +(that: CoordinateDataScores): CoordinateDataScores = fullOuterJoin((a, b) => a + b, that)

  /**
   * The minus operation for [[CoordinateDataScores]].
   *
   * @note This operation performs a full outer join.
   * @param that The [[CoordinateDataScores]] instance to subtract from this instance
   * @return A new [[CoordinateDataScores]] instance encapsulating the subtracted values
   */
  override def -(that: CoordinateDataScores): CoordinateDataScores = fullOuterJoin((a, b) => a - b, that)

  /**
   * Compare two [[CoordinateDataScores]]s objects.
   *
   * @param that Some other object
   * @return True if the both [[CoordinateDataScores]] objects have identical scores for each unique ID, false otherwise
   */
  override def equals(that: Any): Boolean = {
    that match {
      case other: CoordinateDataScores =>
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
