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
package com.linkedin.photon.ml.spark

import scala.reflect.ClassTag

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.{RDDInfo, StorageLevel}

/**
 * A trait containing simple operations on [[RDD]]s.
 */
protected[ml] trait RDDLike {

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  def sparkContext: SparkContext

  /**
   * Assign a given name to all [[RDD]]s in this object.
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the names of all of its [[RDD]]s assigned
   */
  def setName(name: String): RDDLike

  /**
   * Set the storage level of all [[RDD]]s in this object, and persist their values across the cluster the first time
   * they are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of all of its [[RDD]]s set
   */
  def persistRDD(storageLevel: StorageLevel): RDDLike

  /**
   * Mark all [[RDD]]s in this object as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @return This object with all of its [[RDD]]s marked non-persistent
   */
  def unpersistRDD(): RDDLike

  /**
   * Materialize all the [[RDD]]s (Spark [[RDD]]s are lazy evaluated: this method forces them to be evaluated).
   *
   * @return This object with all of its [[RDD]]s materialized
   */
  def materialize(): RDDLike

  /**
   * Materialize the given [[RDD]]s, if they are not already materialized and cached. Forcing an [[RDD]] to be evaluated
   * is a waste of resources if it is already materialized.
   *
   * @tparam T Generic type to handle any [[RDD]]
   * @param rdds The [[RDD]]s to materialize
   */
  protected def materializeOnce[T <: RDD[_] : ClassTag](rdds: T*): Unit = {
    val idSet = sparkContext.getRDDStorageInfo.map(_.id).toSet

    rdds.foreach { rdd =>
      if (!idSet.contains(rdd.id)) rdd.count()
    }
  }
}
