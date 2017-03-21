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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.spark.RDDLike

/**
 * A base class for tracking scored data points, where the scores are stored in an [[RDD]] which associates the unique
 * ID of a data point with a score object.
 *
 * @param scores Data point scores, as described above
 */
abstract protected[ml] class DataScores[T, D <: DataScores[T, D]](scores: RDD[(Long, T)]) extends RDDLike {

  /**
   * Get the Spark context for the distributed scores.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = scores.sparkContext

  /**
   * Set the name of [[scores]].
   *
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the name of [[scores]] assigned
   */
  override def setName(name: String): RDDLike = {
    scores.setName(name)
    this
  }

  /**
   * Set the storage level of [[scores]].
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[scores]] set
   */
  override def persistRDD(storageLevel: StorageLevel): RDDLike = {
    if (!scores.getStorageLevel.isValid) scores.persist(storageLevel)
    this
  }

  /**
   * Mark [[scores]] as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @return This object with [[scores]] marked non-persistent
   */
  override def unpersistRDD(): RDDLike = {
    if (scores.getStorageLevel.isValid) scores.unpersist()
    this
  }

  /**
   * Materialize [[scores]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[scores]] materialized
   */
  override def materialize(): RDDLike = {
    scores.count()
    this
  }

  /**
   * The addition operation for [[DataScores]].
   *
   * @note This operation performs a full outer join.
   * @param that The [[DataScores]] instance to add to this instance
   * @return A new [[DataScores]] instance encapsulating the accumulated values
   */
  def +(that: D): D

  /**
   * The minus operation for [[DataScores]].
   *
   * @note This operation performs a full outer join.
   * @param that The [[DataScores]] instance to subtract from this instance
   * @return A new [[DataScores]] instance encapsulating the subtracted values
   */
  def -(that: D): D
}
