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

import breeze.linalg.Vector
import org.testng.Assert.{assertEquals, assertTrue}
import org.testng.annotations.Test

import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.{DidNotConverge, FunctionValuesConverged, GradientConverged}

/**
 * Integration tests for [[RandomEffectOptimizationTracker]].
 */
class RandomEffectOptimizationTrackerIntegTest extends SparkTestUtils {

  /**
   *
   */
  @Test
  def testApply(): Unit = sparkTest("testApply") {

    // Tracker #1:
    // Convergence Reason = Function Values Converged
    // # Optimization States = 1
    val optimizationStatesTracker1 = new OptimizationStatesTracker()
    optimizationStatesTracker1.convergenceReason = Some(FunctionValuesConverged)
    RandomEffectOptimizationTrackerIntegTest
      .getDummyOptimizerStates(1)
      .foreach(optimizationStatesTracker1.track)

    // Tracker #2:
    // Convergence Reason = Gradient Converged
    // # Optimization States = 2
    val optimizationStatesTracker2 = new OptimizationStatesTracker()
    optimizationStatesTracker2.convergenceReason = Some(GradientConverged)
    RandomEffectOptimizationTrackerIntegTest
      .getDummyOptimizerStates(2)
      .foreach(optimizationStatesTracker2.track)

    // Tracker #3:
    // Convergence Reason = None
    // # Optimization States = 0
    val optimizationStatesTracker3 = new OptimizationStatesTracker()

    // Create RDD of OptimizationStateTrackers objects, and use it to create RandomEffectOptimizationStateTracker
    val statesTrackerRdd =
      sc.parallelize(Seq(optimizationStatesTracker1, optimizationStatesTracker2, optimizationStatesTracker3))
    val randomEffectOptimizationTracker = RandomEffectOptimizationTracker(statesTrackerRdd)

    // Test RandomEffectOptimizationStateTracker members match manually computed values
    val convergenceReasons = randomEffectOptimizationTracker.convergenceReasons
    val iterationStats = randomEffectOptimizationTracker.iterationsStats
    val timeElapsedStats = randomEffectOptimizationTracker.timeElapsedStats

    assertEquals(convergenceReasons.size, 3)
    assertTrue(convergenceReasons.contains(FunctionValuesConverged))
    assertEquals(convergenceReasons(FunctionValuesConverged), 1)
    assertTrue(convergenceReasons.contains(GradientConverged))
    assertEquals(convergenceReasons(GradientConverged), 1)
    assertTrue(convergenceReasons.contains(DidNotConverge))
    assertEquals(convergenceReasons(DidNotConverge), 1)

    assertEquals(iterationStats.count, 3)
    assertEquals(iterationStats.min, 0D)
    assertEquals(iterationStats.max, 2D)
    assertEquals(iterationStats.mean, 1D)
    assertEquals(iterationStats.sampleVariance, 1D)

    assertEquals(timeElapsedStats.count, 2)
  }
}

object RandomEffectOptimizationTrackerIntegTest {

  /**
   * Generate dummy [[OptimizerState]] objects.
   *
   * @param numStates Number of states to generate
   * @return One or more dummy [[OptimizerState]] objects
   */
  def getDummyOptimizerStates(numStates: Int): Seq[OptimizerState] = {
    val coefficients = Vector.zeros[Double](0)
    val value = 0.0
    val gradient = Vector.zeros[Double](0)
    (1 to numStates).map(iter => OptimizerState(coefficients, value, gradient, iter))
  }
}
