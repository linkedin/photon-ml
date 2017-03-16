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
package com.linkedin.photon.ml.data

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.spark.RDDLike

/**
  * The scores used throughout [[com.linkedin.photon.ml.algorithm.CoordinateDescent]], in order to carry on both the
  * offsets and residuals computed during each iteration. In the current implementation, the scores are of form
  * [[RDD]][([[Long]], [[ScoredGameDatum]])], where the [[Long]] typed variable represents the unique ID of each data
  * point, while the [[ScoredGameDatum]] typed variable represents the scored example.
  *
  * @param scores The scores consist of (unique ID, score) pairs as explained above.
  */
protected[ml] class KeyValueScore(val scores: RDD[(Long, ScoredGameDatum)]) extends RDDLike {

  /**
   * Get the Spark context for the distributed scores.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = scores.sparkContext

  /**
   * Set the name of the scores [[RDD]].
   *
   * @param name The parent name for all RDDs in this class
   * @return This object with all its RDDs' name assigned
   */
  override def setName(name: String): KeyValueScore = {
    scores.setName(name)
    this
  }

  /**
   * Set the storage level of the scores [[RDD]]
   *
   * @param storageLevel The storage level
   * @return This object with all its RDDs' storage level set
   */
  override def persistRDD(storageLevel: StorageLevel): KeyValueScore = {
    if (!scores.getStorageLevel.isValid) scores.persist(storageLevel)
    this
  }

  /**
   * Mark the scores [[RDD]] as no longer needed.
   *
   * @return This object with all its RDDs unpersisted
   */
  override def unpersistRDD(): KeyValueScore = {
    if (scores.getStorageLevel.isValid) scores.unpersist()
    this
  }

  /**
   * Materialize the scores [[RDD]] (Spark [[RDD]] objects are lazy-computed: this forces computation).
   *
   * @return This object with all its RDDs materialized
   */
  override def materialize(): KeyValueScore = {
    scores.count()
    this
  }

  /**
   * Generic method to combine two [[KeyValueScore]] objects.
   *
   * @param op The operator to combine two [[ScoredGameDatum]]
   * @param that The [[KeyValueScore]] to merge with this instance
   * @return A merged [[KeyValueScore]]
   */
  private def fullOuterJoin(op: (ScoredGameDatum, ScoredGameDatum) => ScoredGameDatum, that: KeyValueScore): KeyValueScore =
    new KeyValueScore(
      this
        .scores
        .cogroup(that.scores)
        .flatMapValues {
          case (Seq(sd1), Seq(sd2)) => Seq(op(sd1, sd2))
          case (Seq(), Seq(sd2)) => Seq(op(sd2.getZeroScoreDatum, sd2))
          case (Seq(sd1), Seq()) => Seq(op(sd1, sd1.getZeroScoreDatum))
        })

  /**
   * The plus operation for the key-value scores.
   *
   * @note This operation performs a full outer join.
   *
   * @param that The other key value score instance
   * @return A new key value score instance encapsulating the accumulated values
   */
  def +(that: KeyValueScore): KeyValueScore = fullOuterJoin((a, b) => a.copy(score = a.score + b.score), that)

  /**
   * The minus operation for the key-value scores.
   *
   * @note This operation performs a full outer join.
   *
   * @param that The other key value score instance
   * @return A new key value score instance encapsulating the subtracted values
   */
  def -(that: KeyValueScore): KeyValueScore = fullOuterJoin((a, b) => a.copy(score = a.score - b.score), that)

  /**
   * Check if two [[KeyValueScore]]s are equal (if they have the [[ScoredGameDatum]] objects for each unique ID, and
   * those objects are equal).
   *
   * @param that Some other object
   * @return True if the input is a [[KeyValueScore]] with identical [[ScoredGameDatum]] for each unique ID, false
   *         otherwise
   */
  override def equals(that: Any): Boolean = {
    that match {
      case other: KeyValueScore =>
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
