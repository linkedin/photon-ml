package com.linkedin.photon.ml

import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel


/**
 * A trait to hold some simple operations on the RDDs
 * @author xazhang
 */
trait RDDLike {

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
  def setName(name: String): this.type

  /**
   * Set the storage level for all RDDs in this class, and to persist their values across operations after the first
   * time it is computed. This can only be used to assign a new storage level if the RDD does not
   * have a storage level set yet.
   * @param storageLevel The storage level
   * @return This object with all its RDDs' storage level set
   */
  def persistRDD(storageLevel: StorageLevel): this.type

  /**
   * Mark the all RDDs as non-persistent, and remove all blocks for them from memory and disk
   * @return This object with all its RDDs unpersisted
   */
  def unpersistRDD(): this.type

  /**
   * Materialize all the RDDs
   * @return This object with all its RDDs materialized
   */
  def materialize(): this.type
}
