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
package com.linkedin.photon.ml

import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel


/**
 * A trait to hold some simple operations on the RDDs
 */
protected[ml] trait RDDLike {

  /**
   * Get the Spark context
   * @return The Sparks context
   */
  def sparkContext: SparkContext

  /**
   * Assign the name for all RDDs in this class
   * @param name The parent name for all RDDs in this class
   * @return This object with all its RDDs' name assigned
   */
  def setName(name: String): RDDLike

  /**
   * Set the storage level for all RDDs in this class, and to persist their values across operations after the first
   * time it is computed. This can only be used to assign a new storage level if the RDD does not
   * have a storage level set yet.
   * @param storageLevel The storage level
   * @return This object with all its RDDs' storage level set
   */
  def persistRDD(storageLevel: StorageLevel): RDDLike

  /**
   * Mark the all RDDs as non-persistent, and remove all blocks for them from memory and disk
   * @return This object with all its RDDs unpersisted
   */
  def unpersistRDD(): RDDLike

  /**
   * Materialize all the RDDs
   * @return This object with all its RDDs materialized
   */
  def materialize(): RDDLike
}
