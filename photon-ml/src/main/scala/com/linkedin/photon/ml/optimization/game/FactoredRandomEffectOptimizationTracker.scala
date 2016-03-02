package com.linkedin.photon.ml.optimization.game

import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.RDDLike

/**
 * Optimization tracker for factored randon effects
 *
 * @param optimizationStatesTrackers state trackers for random effect and fixed effect problems
 * @author xazhang
 */
protected[ml] class FactoredRandomEffectOptimizationTracker(
    optimizationStatesTrackers: Array[(RandomEffectOptimizationTracker, FixedEffectOptimizationTracker)])
  extends OptimizationTracker with RDDLike {

  override def sparkContext: SparkContext = {
    assert(optimizationStatesTrackers.nonEmpty, "optimizationStatesTrackers is empty!")
    optimizationStatesTrackers.head._1.sparkContext
  }

  override def setName(name: String): this.type = {
    optimizationStatesTrackers.zipWithIndex.foreach { case ((rdd, _), idx) =>
        rdd.setName(s"$name: $idx")
    }
    this
  }

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    optimizationStatesTrackers.foreach(_._1.persistRDD(storageLevel))
    this
  }

  override def unpersistRDD(): this.type = {
    optimizationStatesTrackers.foreach(_._1.unpersistRDD())
    this
  }

  override def materialize(): this.type = {
    optimizationStatesTrackers.foreach(_._1.materialize())
    this
  }

  /**
   * Build a summary string for the tracker
   *
   * @return string representation
   */
  override def toSummaryString: String = {
    optimizationStatesTrackers.zipWithIndex
        .map { case ((randomEffectOptimizationTracker, fixedEffectOptimizationTracker), idx) =>
      s"Idx: $idx\nrandomEffectOptimizationTracker:${randomEffectOptimizationTracker.toSummaryString}\n" +
          s"fixedEffectOptimizationTracker:${fixedEffectOptimizationTracker.toSummaryString}\n"
    }.mkString("\n")
  }
}

object FactoredRandomEffectOptimizationTracker {

}
