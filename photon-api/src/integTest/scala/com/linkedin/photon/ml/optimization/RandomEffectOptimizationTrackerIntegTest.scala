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

import breeze.linalg.Vector
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.FunctionValuesConverged

/**
 * Some simple tests for random effect optimization tracker.
 */
class RandomEffectOptimizationTrackerIntegTest extends SparkTestUtils {

  @Test
  def testCountConvergenceReasons(): Unit = sparkTest("testCountConvergenceReasons") {
    val optimizationStatesTracker1 = new OptimizationStatesTracker()
    optimizationStatesTracker1.convergenceReason = Some(FunctionValuesConverged)
    val optimizationStatesTracker2 = new OptimizationStatesTracker()
    val optimizationStatesTrackerAsRDD = sc.parallelize(Seq(optimizationStatesTracker1, optimizationStatesTracker2))
    val randomEffectOptimizationTracker = new RandomEffectOptimizationTracker(optimizationStatesTrackerAsRDD)
    assertEquals(randomEffectOptimizationTracker.countConvergenceReasons,
      Map(FunctionValuesConverged.reason -> 1, RandomEffectOptimizationTracker.NOT_CONVERGED ->1))
  }

  @Test
  def testGetNumIterationStats(): Unit = sparkTest("testGetNumIterationStats") {
    val emptyOptimizationStatesTracker = new OptimizationStatesTracker()
    val normalOptimizationStatesTracker = new OptimizationStatesTracker()
    val numStates = 5
    RandomEffectOptimizationTrackerIntegTest.getDummyOptimizerStates(numStates)
        .foreach(normalOptimizationStatesTracker.track)
    val optimizationStatesTrackerAsRDD =
      sc.parallelize(Seq(emptyOptimizationStatesTracker, normalOptimizationStatesTracker))
    val randomEffectOptimizationTracker = new RandomEffectOptimizationTracker(optimizationStatesTrackerAsRDD)
    val statCounter = sc.parallelize(Seq(0, numStates)).stats()
    assertEquals(randomEffectOptimizationTracker.getNumIterationStats.toString(), statCounter.toString())
  }

  @Test
  def testGetElapsedTimeStats(): Unit = sparkTest("testGetNumIterationStats") {
    // here we only test the empty case that PR https://github.com/linkedin/photon-ml/pull/115 is trying to address

    val emptyOptimizationStatesTracker1 = new OptimizationStatesTracker()
    val emptyOptimizationStatesTracker2 = new OptimizationStatesTracker()
    val optimizationStatesTrackerAsRDD =
      sc.parallelize(Seq(emptyOptimizationStatesTracker1, emptyOptimizationStatesTracker2))
    val emptyRandomEffectOptimizationTracker = new RandomEffectOptimizationTracker(optimizationStatesTrackerAsRDD)
    val statCounter = sc.parallelize(Seq(0d, 0d)).stats()
    assertEquals(emptyRandomEffectOptimizationTracker.getNumIterationStats.toString(), statCounter.toString())
  }
}

object RandomEffectOptimizationTrackerIntegTest {

  /**
   *
   * @param numStates
   * @return
   */
  def getDummyOptimizerStates(numStates: Int): Seq[OptimizerState] = {
    val coefficients = Vector.zeros[Double](0)
    val value = 0.0
    val gradient = Vector.zeros[Double](0)
    (0 until numStates).map(iter => OptimizerState(coefficients, value, gradient, iter))
  }
}
