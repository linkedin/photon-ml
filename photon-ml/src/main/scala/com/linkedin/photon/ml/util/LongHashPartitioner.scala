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
package com.linkedin.photon.ml.util


import org.apache.spark.Partitioner


/**
 * @author xazhang
 */
class LongHashPartitioner(partitions: Int) extends Partitioner {
  def getPartition(key: Any): Int = key match {
    case long: Long => (math.abs(long) % partitions).toInt
    case any =>
      throw new IllegalArgumentException(s"Expected key of ${this.getClass} is Long, but ${any.getClass} is found")
  }

  override def equals(other: Any): Boolean = other match {
    case h: LongHashPartitioner => h.numPartitions == numPartitions
    case _ => false
  }

  def numPartitions: Int = partitions

  override def hashCode: Int = numPartitions
}
