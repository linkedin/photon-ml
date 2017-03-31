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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.spark.RDDLike

/**
 * Optimization tracker for factored random effects
 *
 * @param optimizationStatesTrackers state trackers for random effect and fixed effect problems
 */
protected[ml] class FactoredRandomEffectOptimizationTracker(
    optimizationStatesTrackers: Array[(RandomEffectOptimizationTracker, FixedEffectOptimizationTracker)])
  extends OptimizationTracker with RDDLike {

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = {
    assert(optimizationStatesTrackers.nonEmpty, "optimizationStatesTrackers is empty")
    optimizationStatesTrackers.head._1.sparkContext
  }

  /**
   * Assign a given name to all [[RandomEffectOptimizationTracker]] objects.
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   *
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the names of all [[RandomEffectOptimizationTracker]] objects assigned
   */
  override def setName(name: String): FactoredRandomEffectOptimizationTracker = {
    optimizationStatesTrackers.zipWithIndex.foreach { case ((rddLike, _), idx) =>
      rddLike.setName(s"$name: $idx")
    }
    this
  }

  /**
   * Set the storage level of all [[RandomEffectOptimizationTracker]] objects, and persist their values across the
   * cluster the first time they are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of all [[RandomEffectOptimizationTracker]] objects set
   */
  override def persistRDD(storageLevel: StorageLevel): FactoredRandomEffectOptimizationTracker = {
    optimizationStatesTrackers.foreach(_._1.persistRDD(storageLevel))
    this
  }

  /**
   * Mark all [[RandomEffectOptimizationTracker]] objects as non-persistent, and remove all blocks for them from memory
   * and disk.
   *
   * @return This object with all [[RandomEffectOptimizationTracker]] objects marked non-persistent
   */
  override def unpersistRDD(): FactoredRandomEffectOptimizationTracker = {
    optimizationStatesTrackers.foreach(_._1.unpersistRDD())
    this
  }

  /**
   * Materialize all [[RandomEffectOptimizationTracker]] objects (Spark [[RDD]]s are lazy evaluated: this method forces
   * them to be evaluated).
   *
   * @return This object with all [[RandomEffectOptimizationTracker]] objects materialized
   */
  override def materialize(): FactoredRandomEffectOptimizationTracker = {
    optimizationStatesTrackers.foreach(_._1.materialize())
    this
  }

  /**
   * Build a summary string for this [[FactoredRandomEffectOptimizationTracker]].
   *
   * @return A summary of this [[FactoredRandomEffectOptimizationTracker]] in string representation
   */
  override def toSummaryString: String =
    optimizationStatesTrackers
      .zipWithIndex
      .map { case ((randomEffectOptimizationTracker, fixedEffectOptimizationTracker), idx) =>
        s"Idx: $idx\nrandomEffectOptimizationTracker:${randomEffectOptimizationTracker.toSummaryString}\n" +
            s"fixedEffectOptimizationTracker:${fixedEffectOptimizationTracker.toSummaryString}\n"
      }
      .mkString("\n")
}
