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
package com.linkedin.photon.ml.util

import org.apache.spark.Partitioner

/**
 * The partitioner for [[Long]] typed keys with given number of partitions.
 *
 * @param partitions The total number of partitions available to the partitioner
 */
protected[ml] class LongHashPartitioner(partitions: Int) extends Partitioner {

  /**
   * Get the number of partitions for this [[LongHashPartitioner]].
   *
   * @return The number of partitions
   */
  override def numPartitions: Int = partitions

  /**
   * Map a key to a partition.
   *
   * @param key The key
   * @return The corresponding partition for the given key
   */
  override def getPartition(key: Any): Int = key match {
    case long: Long => (math.abs(long) % partitions).toInt
    case any =>
      throw new IllegalArgumentException(s"Expected key of ${this.getClass} is Long, but ${any.getClass} is found")
  }

  /**
   * Compares two [[LongHashPartitioner]] objects.
   *
   * @param that Some other object
   * @return True if both [[LongHashPartitioner]] split between the same number of partitions, false otherwise
   */
  override def equals(that: Any): Boolean = that match {
    case other: LongHashPartitioner => this.numPartitions == other.numPartitions
    case _ => false
  }

  /**
   * Returns a hash code value for the object.
   *
   * @return An [[Int]] hash code
   */
  override def hashCode: Int = numPartitions
}
