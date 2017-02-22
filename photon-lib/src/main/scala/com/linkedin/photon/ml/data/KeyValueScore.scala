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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.spark.RDDLike

/**
 * The scores used throughout [[com.linkedin.photon.ml.algorithm.CoordinateDescent]], in order to carry on both the
 * offsets and residuals computed during each iteration. In the current implementation, the scores are of form
 * [[RDD]][([[Long]], [[Double]])], where the [[Long]] typed variable represents the unique ID of each data point,
 * while the [[Double]] typed variable represents the score.
 *
 * @param scores the scores consists of (unique ID, score) pairs as explained above.
 */
protected[ml] class KeyValueScore(val scores: RDD[(Long, Double)]) extends RDDLike {

  /**
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = scores.sparkContext

  /**
   *
   * @param name The parent name for all RDDs in this class
   * @return This object with all its RDDs' name assigned
   */
  override def setName(name: String): KeyValueScore = {
    scores.setName(name)
    this
  }

  /**
   *
   * @param storageLevel The storage level
   * @return This object with all its RDDs' storage level set
   */
  override def persistRDD(storageLevel: StorageLevel): KeyValueScore = {
    if (!scores.getStorageLevel.isValid) scores.persist(storageLevel)
    this
  }

  /**
   *
   * @return This object with all its RDDs unpersisted
   */
  override def unpersistRDD(): KeyValueScore = {
    if (scores.getStorageLevel.isValid) scores.unpersist()
    this
  }

  /**
   *
   * @return This object with all its RDDs materialized
   */
  override def materialize(): KeyValueScore = {
    scores.count()
    this
  }

  /**
   * The plus operation for the key value scores.
   *
   * @param that The other key value score instance
   * @return A new key value score instance encapsulating the accumulated values
   */
  def +(that: KeyValueScore): KeyValueScore = {
    val addedScores =
      this.scores
        .fullOuterJoin(that.scores)
        .mapValues { case (thisScore, thatScore) => thisScore.getOrElse(0.0) + thatScore.getOrElse(0.0) }
    new KeyValueScore(addedScores)
  }

  /**
   * The minus operation for the key value scores.
   *
   * @param that The other key value score instance
   * @return A new key value score instance encapsulating the subtracted values
   */
  def -(that: KeyValueScore): KeyValueScore = {
    val subtractedScores =
      this.scores
        .fullOuterJoin(that.scores)
        .mapValues { case (thisScore, thatScore) => thisScore.getOrElse(0.0) - thatScore.getOrElse(0.0) }
    new KeyValueScore(subtractedScores)
  }

  /**
   *
   * @param that
   * @return
   */
  override def equals(that: Any): Boolean = {
    that match {
      case other: KeyValueScore => this.scores.fullOuterJoin(other.scores).mapPartitions(iterator =>
        Iterator.single(iterator.forall { case (_, (thisScoreOpt1, thisScoreOpt2)) =>
          thisScoreOpt1.isDefined && thisScoreOpt2.isDefined &&
            math.abs(thisScoreOpt1.get - thisScoreOpt2.get) < MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD
        })
      ).filter(!_).count() == 0
      case _ => false
    }
  }
}
