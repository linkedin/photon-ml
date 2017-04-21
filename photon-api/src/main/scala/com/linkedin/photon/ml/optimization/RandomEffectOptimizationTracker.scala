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

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = optimizationStatesTrackers.sparkContext

  /**
   * Assign a given name to [[optimizationStatesTrackers]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   *
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the name of [[optimizationStatesTrackers]] assigned
   */
  override def setName(name: String): RandomEffectOptimizationTracker = {

    optimizationStatesTrackers.setName(name)

    this
  }

  /**
   * Set the storage level of [[optimizationStatesTrackers]], and persist their values across the cluster the first time
   * they are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[optimizationStatesTrackers]] set
   */
  override def persistRDD(storageLevel: StorageLevel): RandomEffectOptimizationTracker = {

    if (!optimizationStatesTrackers.getStorageLevel.isValid) optimizationStatesTrackers.persist(storageLevel)

    this
  }

  /**
   * Mark [[optimizationStatesTrackers]] as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @return This object with [[optimizationStatesTrackers]] marked non-persistent
   */
  override def unpersistRDD(): RandomEffectOptimizationTracker = {

    if (optimizationStatesTrackers.getStorageLevel.isValid) optimizationStatesTrackers.unpersist()

    this
  }

  /**
   * Materialize [[optimizationStatesTrackers]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be
   * evaluated).
   *
   * @return This object with [[optimizationStatesTrackers]] materialized
   */
  override def materialize(): RandomEffectOptimizationTracker = {

    materializeOnce(optimizationStatesTrackers)

    this
  }

  /**
   * Count the number of occurrence of all returned convergence reasons
   *
   * @return A map of convergence reason to its count of occurrence from the random effect optimization problems
   */
  def countConvergenceReasons: Map[String, Int] =

    optimizationStatesTrackers
      .map { optimizationStatesTracker =>
        (optimizationStatesTracker
          .convergenceReason
          .map(_.reason)
          .getOrElse(RandomEffectOptimizationTracker.NOT_CONVERGED), 1)
      }
      .reduceByKey(_ + _)
      .collectAsMap()

  /**
   * Get stats counter of the number of iterations tracked in all optimization states trackers
   *
   * @return A [[StatCounter]] of the number of iterations tracked in all optimization states trackers
   */
  def getNumIterationStats: StatCounter = optimizationStatesTrackers.map(_.getTrackedStates.length).stats()

  /**
   * Get stats counter on the elapsed time tracked in all optimization states trackers
   *
   * @return A [[StatCounter]] of the elapsed time tracked in all optimization states trackers
   */
  def getElapsedTimeStats: StatCounter =

    optimizationStatesTrackers
      .map { optimizationStatesTracker =>
        val timeHistory = optimizationStatesTracker.getTrackedTimeHistory

        if (timeHistory.length > 0) {
          timeHistory.last * 1e-6
        } else {
          0.0
        }
      }
      .stats()

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
