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

import org.apache.spark.SparkContext
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
