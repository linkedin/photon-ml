package com.linkedin.photon.ml.data

import scala.collection.{Map, immutable, mutable}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{HashPartitioner, Partitioner}
import org.apache.spark.rdd.RDD

/**
 * Spark partitioner implementation for random effect datasets
 *
 * @param idToPartitionMap random effect id to partition map
 * @author xazhang
 */
class RandomEffectIdPartitioner(idToPartitionMap: Broadcast[Map[String, Int]]) extends Partitioner {

  val partitions = idToPartitionMap.value.values.max + 1

  def numPartitions: Int = partitions

  override def equals(other: Any): Boolean = other match {
    case rep: RandomEffectIdPartitioner =>
      idToPartitionMap.value.forall { case (key, partition) => rep.getPartition(key) == partition }
    case _ => false
  }

  def getPartition(key: Any): Int = key match {
    case string: String =>
      idToPartitionMap.value.getOrElse(string, defaultPartitioner.getPartition(string))
    case any =>
      throw new IllegalArgumentException(s"Expected key of ${this.getClass} is String, but ${any.getClass} found")
  }

  def defaultPartitioner = new HashPartitioner(partitions)

  override def hashCode: Int = idToPartitionMap.hashCode()
}

object RandomEffectIdPartitioner {

  /**
   * Generate a partitioner from the random effect dataset
   *
   * @param numPartitions number of partitions
   * @param randomEffectId the random effect type id (e.g. "memberId")
   * @param gameDataSet the dataset
   * @param partitionerCapacity partitioner capacity
   * @return the partitioner
   */
  def generateRandomEffectIdPartitionerFromGameDataSet(
      numPartitions: Int,
      randomEffectId: String,
      gameDataSet: RDD[(Long, GameData)],
      partitionerCapacity: Int = 10000): RandomEffectIdPartitioner = {

    assert(numPartitions > 0, s"Number of partitions ($numPartitions) has to be larger than 0.")
    val sortedRandomEffectIds =
      gameDataSet
          .values
          .filter(_.randomEffectIdToIndividualIdMap.contains(randomEffectId))
          .map(gameData => (gameData.randomEffectIdToIndividualIdMap(randomEffectId), 1))
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

    sortedRandomEffectIds.foreach { case (id, size) =>
      val (partition, currentSize) = minHeap.dequeue()
      idToPartitionMapBuilder += id -> partition
      minHeap.enqueue((partition, currentSize + size))
    }

    new RandomEffectIdPartitioner(gameDataSet.sparkContext.broadcast(idToPartitionMapBuilder.result()))
  }
}
