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

import scala.collection.{Map, immutable, mutable}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{HashPartitioner, Partitioner}
import org.apache.spark.rdd.RDD

/**
 * Spark partitioner implementation for random effect datasets, that takes the imbalanced data size across different
 * random effects into account (e.g., a popular item may be associated with more data points than a less popular one)
 *
 * @param idToPartitionMap random effect type to partition map
 */
protected[ml] class RandomEffectDataSetPartitioner(idToPartitionMap: Broadcast[Map[String, Int]]) extends Partitioner {

  val partitions = idToPartitionMap.value.values.max + 1

  def numPartitions: Int = partitions

  override def equals(other: Any): Boolean = other match {
    case rep: RandomEffectDataSetPartitioner =>
      idToPartitionMap.value.forall { case (key, partition) => rep.getPartition(key) == partition }
    case _ => false
  }

  def getPartition(key: Any): Int = key match {
    case string: String =>
      idToPartitionMap.value.getOrElse(string, defaultPartitioner.getPartition(string))
    case any =>
      throw new IllegalArgumentException(s"Expected key of ${this.getClass} is String, but ${any.getClass} found")
  }

  def defaultPartitioner: HashPartitioner = new HashPartitioner(partitions)

  override def hashCode: Int = idToPartitionMap.hashCode()
}

object RandomEffectDataSetPartitioner {

  /**
   * Generate a partitioner from the random effect dataset
   *
   * @param numPartitions number of partitions
   * @param randomEffectType the random effect type (e.g. "memberId")
   * @param gameDataSet the dataset
   * @param partitionerCapacity partitioner capacity
   * @return the partitioner
   */
  def generateRandomEffectDataSetPartitionerFromGameDataSet(
      numPartitions: Int,
      randomEffectType: String,
      gameDataSet: RDD[(Long, GameDatum)],
      partitionerCapacity: Int = 10000): RandomEffectDataSetPartitioner = {

    assert(numPartitions > 0, s"Number of partitions ($numPartitions) has to be larger than 0.")
    val sortedRandomEffectTypes =
      gameDataSet
          .values
          .filter(_.idTypeToValueMap.contains(randomEffectType))
          .map(gameData => (gameData.idTypeToValueMap(randomEffectType), 1))
          .reduceByKey(_ + _)
          .collect()
          .sortBy(_._2 * -1)
          .take(partitionerCapacity)

    val ordering = new Ordering[(Int, Int)] {
      def compare(pair1: (Int, Int), pair2: (Int, Int)) = pair2._2 compare pair1._2
    }

    val minHeap = mutable.PriorityQueue.newBuilder[(Int, Int)](ordering)
    minHeap ++= Array.tabulate[(Int, Int)](numPartitions)(i => (i, 0))
    val idToPartitionMapBuilder = immutable.Map.newBuilder[String, Int]
    idToPartitionMapBuilder.sizeHint(numPartitions)

    sortedRandomEffectTypes.foreach { case (id, size) =>
      val (partition, currentSize) = minHeap.dequeue()
      idToPartitionMapBuilder += id -> partition
      minHeap.enqueue((partition, currentSize + size))
    }

    new RandomEffectDataSetPartitioner(gameDataSet.sparkContext.broadcast(idToPartitionMapBuilder.result()))
  }
}
