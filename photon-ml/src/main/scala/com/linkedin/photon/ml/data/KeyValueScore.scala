package com.linkedin.photon.ml.data

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.RDDLike

/**
 * Key value score accumulator representation
 *
 * @param scores the scores
 * @author xazhang
 */
class KeyValueScore(val scores: RDD[(Long, Double)]) extends RDDLike {

  override def sparkContext: SparkContext = scores.sparkContext

  override def setName(name: String): this.type = {
    scores.setName(name)
    this
  }

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!scores.getStorageLevel.isValid) scores.persist(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    if (scores.getStorageLevel.isValid) scores.unpersist()
    this
  }

  override def materialize(): this.type = {
    scores.count()
    this
  }

  /**
   * Accumulate key value score values
   *
   * @param that the other key value score instance
   * @return a new key value score instance encapsulating the accumulated values
   */
  def +(that: KeyValueScore): KeyValueScore = {
    val addedScores =
      this.scores
        .fullOuterJoin(that.scores)
        .mapValues { case (thisScore, thatScore) => thisScore.getOrElse(0.0) + thatScore.getOrElse(0.0) }
    new KeyValueScore(addedScores)
  }

  /**
   * Remove key value score values
   *
   * @param that the other key value score instance
   * @return a new key value score instance encapsulating the accumulated values
   */
  def -(that: KeyValueScore): KeyValueScore = {
    val subtractedScores =
      this.scores
        .fullOuterJoin(that.scores)
        .mapValues { case (thisScore, thatScore) => thisScore.getOrElse(0.0) - thatScore.getOrElse(0.0) }
    new KeyValueScore(subtractedScores)
  }
}
