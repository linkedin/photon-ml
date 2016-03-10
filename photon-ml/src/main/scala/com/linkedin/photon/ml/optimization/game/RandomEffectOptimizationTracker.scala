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
