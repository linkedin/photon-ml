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
