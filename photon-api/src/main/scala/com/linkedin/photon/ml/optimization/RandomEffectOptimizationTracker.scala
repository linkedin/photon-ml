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

import org.apache.spark.rdd.RDD
import org.apache.spark.util.StatCounter

import com.linkedin.photon.ml.util.{ConvergenceReason, DidNotConverge}

/**
 * Optimization tracker for a [[com.linkedin.photon.ml.optimization.game.RandomEffectOptimizationProblem]].
 *
 * @param convergenceReasons A [[Map]] of [[ConvergenceReason]] and the number of per-entity models which halted
 *                           optimization for that reason
 * @param iterationsStats Statistical summary of the number of iterations required per-entity to converge
 * @param timeElapsedStats Statistical summary of the time elapsed per-entity to converge
 */
protected[ml] class RandomEffectOptimizationTracker(
    val convergenceReasons: Map[ConvergenceReason, Int],
    val iterationsStats: StatCounter,
    val timeElapsedStats: StatCounter)
  extends OptimizationTracker {

  /**
   * Build a well-formatted summary string for output to logs.
   *
   * @return Summary of this [[RandomEffectOptimizationTracker]] in [[String]] format
   */
  override def toSummaryString: String =
    RandomEffectOptimizationTracker.SUMMARY_FORMAT.format(convergenceReasons, iterationsStats, timeElapsedStats)
}

object RandomEffectOptimizationTracker{

  protected[optimization] val NOT_CONVERGED = "Not converged"
  protected[optimization] val SUMMARY_FORMAT =
    """Convergence reasons stats: %s
      |Number of iterations stats: %s
      |Time elapsed stats: %s""".stripMargin

  /**
   * Helper method to generate a [[RandomEffectOptimizationTracker]] from a [[RDD]] of [[OptimizationStatesTracker]]
   * received from a [[com.linkedin.photon.ml.optimization.game.RandomEffectOptimizationProblem]].
   *
   * @param optimizationStatesTrackers A [[RDD]] of [[OptimizationStatesTracker]]
   * @return A new [[RandomEffectOptimizationTracker]]
   */
  def apply(optimizationStatesTrackers: RDD[OptimizationStatesTracker]): RandomEffectOptimizationTracker = {

    val convergenceReasons = optimizationStatesTrackers
      .map { optimizationStatesTracker =>
        (optimizationStatesTracker.convergenceReason.getOrElse(DidNotConverge), 1)
      }
      .reduceByKey(_ + _)
      .collectAsMap()
      .toMap
    val iterationsStats = optimizationStatesTrackers
      .map { optimizationStatesTracker =>
        val trackedStates = optimizationStatesTracker.getTrackedStates

        if (trackedStates.isEmpty) {
          0
        } else {
          trackedStates.last.iter
        }
      }
      .stats()
    val timeElapsedStats = optimizationStatesTrackers
      .map(_.getTrackedTimeHistory)
      // Filter out times for random effects that never ran
      .filter(_.length > 0)
      .map(_.last * 1E-3)
      .stats()

    new RandomEffectOptimizationTracker(convergenceReasons, iterationsStats, timeElapsedStats)
  }
}
