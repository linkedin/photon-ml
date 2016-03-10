package com.linkedin.photon.ml.optimization.game

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.optimization._

/**
 * Optimization tracker for random effect optimization problems
 *
 * @param optimizationStatesTrackers state trackers for the inidividual optimization problems
 * @author xazhang
 */
class RandomEffectOptimizationTracker(val optimizationStatesTrackers: RDD[OptimizationStatesTracker])
  extends OptimizationTracker with RDDLike {

  override def sparkContext: SparkContext = optimizationStatesTrackers.sparkContext

  override def setName(name: String): this.type = {
    optimizationStatesTrackers.setName(name)
    this
  }

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!optimizationStatesTrackers.getStorageLevel.isValid) optimizationStatesTrackers.persist(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    if (optimizationStatesTrackers.getStorageLevel.isValid) optimizationStatesTrackers.unpersist()
    this
  }

  override def materialize(): this.type = {
    optimizationStatesTrackers.count()
    this
  }

  /**
   * Build a summary string for the tracker
   *
   * @return string representation
   */
  override def toSummaryString: String = {
    val convergenceReasons = optimizationStatesTrackers.map { optimizationStatesTracker =>
      if (optimizationStatesTracker.convergenceReason.isDefined) {
        (optimizationStatesTracker.convergenceReason.get.reason, 1)
      } else {
        ("Not converged", 1)
      }
    }.reduceByKey(_ + _).collectAsMap().mkString("\n")
    val numIterationsStats = optimizationStatesTrackers.map(_.getTrackedTimeHistory.length).stats()
    val timeElapsedStats = optimizationStatesTrackers.map { optimizationStatesTracker =>
      optimizationStatesTracker.getTrackedTimeHistory.last * 1e-6
    }.stats()
    s"Convergence reasons stats:\n$convergenceReasons\n" +
        s"Number of iterations stats: $numIterationsStats\n" +
        s"Time elapsed stats: $timeElapsedStats"
  }
}
