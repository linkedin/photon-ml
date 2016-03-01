package com.linkedin.photon.ml.constants

import org.apache.spark.storage.{StorageLevel => SparkStorageLevel}

/**
 * @author xazhang
 */
object StorageLevel {
  val FREQUENT_REUSE_RDD_STORAGE_LEVEL = SparkStorageLevel.MEMORY_AND_DISK
  val INFREQUENT_REUSE_RDD_STORAGE_LEVEL = SparkStorageLevel.DISK_ONLY
}
