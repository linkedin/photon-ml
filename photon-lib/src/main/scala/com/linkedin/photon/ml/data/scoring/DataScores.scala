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
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.constants.DataConst

/**
 * A base class for tracking scored data points, where the scores are stored in an [[DataFrame]]
 * which associates the unique
 * ID of a data point with a score object.
 *
 * @param scores Data point scores, as described above
 */
abstract protected[ml] class DataScores[D <: DataScores[D]](
    val scores: DataFrame)
  extends RDDLike {

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

  /**
   * Get the Spark context for the distributed scores.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = SparkSession.builder.getOrCreate.sparkContext

  /* RDDLike methods */
 override def setName(name: String): RDDLike = {

    this
  }

  /**
   * Set the storage level of [[scores]].
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[scores]] set
   */
  override def persistRDD(storageLevel: StorageLevel): RDDLike = {

    scores.persist(storageLevel)

    this
  }

  /**
   * Mark [[scores]] as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @return This object with [[scores]] marked non-persistent
   */
  override def unpersistRDD(): RDDLike = {

    scores.unpersist()

    this
  }

  /**
   * Materialize [[scores]] (Spark data are lazy evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[scores]] materialized
   */
  override def materialize(): RDDLike = {

    scores.count()

    this
  }

  /**
   * Method used to define equality on multiple class levels while conforming to equality contract. Defines under
   * what circumstances this class can equal another class.
   *
   * @param other Some other object
   * @return Whether this object can equal the other object
   */
  def canEqual(other: Any): Boolean = other.isInstanceOf[DataScores[D]]

  /**
   * Compare two [[DataScores]]s objects.
   *
   * @param other Some other object
   * @return True if the both [[DataScores]] objects have identical scores for each unique ID, false otherwise
   */
  override def equals(other: Any): Boolean = other match {

    case that: DataScores[D] =>

      val canEqual = this.canEqual(that)
      lazy val areEqual = this
        .scores
        .withColumnRenamed(DataConst.SCORE, "s1")
        .join(that.scores.withColumnRenamed(DataConst.SCORE, "s2"), col(DataConst.ID), "fullouter")
        .filter("s1 is null or s2 is null or s1 != s2")
        .head(1)
        .isEmpty

      canEqual && areEqual

    case _ =>
      false
  }

  /**
   * Returns a hash code value for the object.
   *
   * @return An [[Int]] hash code
   */
  override def hashCode: Int = scores.hashCode()

}
