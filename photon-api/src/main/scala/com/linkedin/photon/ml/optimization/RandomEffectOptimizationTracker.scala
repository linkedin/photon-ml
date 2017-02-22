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
package com.linkedin.photon.ml.optimization

import scala.collection.Map

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.StatCounter

import com.linkedin.photon.ml.spark.RDDLike

/**
 * Optimization tracker for random effect optimization problems
 *
 * @param optimizationStatesTrackers state trackers for random effect optimization problems
 */
protected[ml] class RandomEffectOptimizationTracker(val optimizationStatesTrackers: RDD[OptimizationStatesTracker])
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
   * Count the number of occurrence of all returned convergence reasons
   *
   * @return A map of convergence reason to its count of occurrence from the random effect optimization problems
   */
  def countConvergenceReasons: Map[String, Int] = {
    optimizationStatesTrackers.map { optimizationStatesTracker =>
      if (optimizationStatesTracker.convergenceReason.isDefined) {
        (optimizationStatesTracker.convergenceReason.get.reason, 1)
      } else {
        (RandomEffectOptimizationTracker.NOT_CONVERGED, 1)
      }
    }.reduceByKey(_ + _).collectAsMap()
  }

  /**
   * Get stats counter of the number of iterations tracked in all optimization states trackers
   *
   * @return A [[StatCounter]] of the number of iterations tracked in all optimization states trackers
   */
  def getNumIterationStats: StatCounter = {
    optimizationStatesTrackers.map(_.getTrackedStates.length).stats()
  }

  /**
   * Get stats counter on the elapsed time tracked in all optimization states trackers
   *
   * @return A [[StatCounter]] of the elapsed time tracked in all optimization states trackers
   */
  def getElapsedTimeStats: StatCounter = {
    optimizationStatesTrackers.map { optimizationStatesTracker =>
      val timeHistory = optimizationStatesTracker.getTrackedTimeHistory
      if (timeHistory.length > 0) {
        timeHistory.last * 1e-6
      } else {
        0.0
      }
    }.stats()
  }

  /**
   * Build a summary string for the tracker
   *
   * @return string representation
   */
  override def toSummaryString: String = {
    val convergenceReasons = countConvergenceReasons
    val numIterationsStats = getNumIterationStats
    val timeElapsedStats = getElapsedTimeStats
    RandomEffectOptimizationTracker.SUMMARY_FORMAT.format(convergenceReasons, numIterationsStats, timeElapsedStats)
  }
}

object RandomEffectOptimizationTracker{
  protected[optimization] val NOT_CONVERGED = "Not converged"

  protected[optimization] val SUMMARY_FORMAT = "Convergence reasons stats:\n%s\nNumber of iterations stats: %s\n" +
    "Time elapsed stats: %s"
}
