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

import scala.collection.{Map, immutable, mutable}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, Partitioner}

import com.linkedin.photon.ml.spark.BroadcastLike

/**
 * Partitioner implementation for random effect datasets, that takes the imbalanced data size across different
 * random effects into account (e.g., a popular item may be associated with more data points than a less popular one).
 *
 * @param idToPartitionMap Random effect type to partition map
 */
protected[ml] class RandomEffectDataSetPartitioner(idToPartitionMap: Broadcast[Map[String, Int]])
  extends Partitioner
  with BroadcastLike {

  val numPartitions: Int = idToPartitionMap.value.values.max + 1

  /**
   *
   * @return This object with all its broadcast variables unpersisted
   */
  override def unpersistBroadcast(): this.type = {
    idToPartitionMap.unpersist()
    this
  }

  /**
   * Compares two RandomEffectDataSetPartitioner.
   *
   * @param other The other RandomEffectDataSetPartitioner to compare with
   * @return true if the two partitioners have the same idToPartitionMap, false otherwise
   */
  override def equals(other: Any): Boolean = other match {
    case rep: RandomEffectDataSetPartitioner =>
      idToPartitionMap.value.forall { case (key, partition) => rep.getPartition(key) == partition }
    case _ => false
  }

  /**
   * Hash code for this partitioner.
   *
   * @return A Int hash code
   */
  override def hashCode: Int = idToPartitionMap.hashCode()

  /**
   * For a given key, get the corresponding partition id. If the key is not in any partition, we randomly assign
   * the training vector to a partition (with Spark's HashPartitioner).
   *
   * @param key A training vector key (String).
   * @return The partition id to which the training vector belongs.
   */
  def getPartition(key: Any): Int = key match {
    case string: String =>
      idToPartitionMap.value.getOrElse(string, new HashPartitioner(numPartitions).getPartition(string))
    case any =>
      throw new IllegalArgumentException(s"Expected key of ${this.getClass} is String, but ${any.getClass} found")
  }
}

object RandomEffectDataSetPartitioner {
  /**
   * Generate a partitioner for one random effect model.
   *
   * A random effect model is composed of multiple sub-models, each trained on data points for a single item.
   * We collect the training vector ids that correspond to the random effect, then build an id to partition map.
   * Data should be distributed across partitions as equally as possible. Since some items have more data points
   * than others, this partitioner uses simple 'bin packing' for distributing data load across partitions (using
   * minHeap).
   *
   * @param numPartitions The number of partitions to fill
   * @param randomEffectType The random effect type (e.g. "memberId")
   * @param gameDataSet The GAME training dataset
   * @param partitionerCapacity The partitioner capacity
   * @return A partitioner for one random effect model
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
      def compare(pair1: (Int, Int), pair2: (Int, Int)): Int = pair2._2 compare pair1._2
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
