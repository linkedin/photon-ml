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

import java.util.Random

import breeze.linalg.{DenseVector, SparseVector, norm}
import breeze.optimize.FirstOrderMinimizer.{FunctionValuesConverged, GradientConverged}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.test.SparkTestUtils
import org.apache.spark.Logging
import org.testng.Assert._
import org.testng.annotations.{Test, DataProvider}


/**
  * Verify that core optimizers do reasonable things on small test problems.
  * @author bdrew
  */
class OptimizerIntegTest extends SparkTestUtils with Logging {
  @DataProvider(parallel = true)
  def optimizeEasyTestFunction(): Array[Array[Object]] = {
    Array(Array(new LBFGS[LabeledPoint]()),
      Array(new TRON[LabeledPoint]()))
  }

  @Test(dataProvider = "optimizeEasyTestFunction")
  def checkEasyTestFunctionLocalNoInitialValue(optim: Optimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]])
  : Unit = {
    optim.setStateTrackingEnabled(true)
    optim.setMaximumIterations(OptimizerIntegTest.MAX_ITERATIONS)
    optim.setTolerance(OptimizerIntegTest.CONVERGENCE_TOLERANCE)
    val features = new SparseVector[Double](Array(), Array(), OptimizerIntegTest.PROBLEM_DIMENSION)
    val pt = new LabeledPoint(label = 1, features, offset = 0, weight = 1)
    optim.optimize(Seq(pt), new IntegTestObjective())
    OptimizerIntegTest.easyOptimizationStatesChecks(optim.getStateTracker.get)
    //test weight point
    val pt2 = new LabeledPoint(label = 1, features, offset = 0, weight = 2.5)
    optim.optimize(Seq(pt2), new IntegTestObjective())
    OptimizerIntegTest.easyOptimizationStatesChecks(optim.getStateTracker.get)
  }

  @Test(dataProvider = "optimizeEasyTestFunction")
  def checkEasyTestFunctionLocalInitialValue(optim: Optimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]]): Unit = {
    optim.setStateTrackingEnabled(true)
    optim.setMaximumIterations(OptimizerIntegTest.MAX_ITERATIONS)
    optim.setTolerance(OptimizerIntegTest.CONVERGENCE_TOLERANCE)
    optim.isReusingPreviousInitialState = false
    val features = new SparseVector[Double](Array(), Array(), OptimizerIntegTest.PROBLEM_DIMENSION)
    val pt = new LabeledPoint(label = 1, features, offset = 0, weight = 1)
    val r = new Random(OptimizerIntegTest.RANDOM_SEED)
    for (iter <- 0 to OptimizerIntegTest.RANDOM_SAMPLES) {
      val initParam = DenseVector.fill[Double](OptimizerIntegTest.PROBLEM_DIMENSION)(r.nextDouble())
      optim.optimize(Array(pt), new IntegTestObjective(), initParam)
      assertTrue(optim.stateTrackingEnabled)
      assertTrue(optim.getStateTracker.isDefined)
      assertTrue(optim.isDone)
      OptimizerIntegTest.easyOptimizationStatesChecks(optim.getStateTracker.get)
    }

    //test weighted sample
    val pt2 = new LabeledPoint(label = 1, features, offset = 0, weight = 10.0)
    for (iter <- 0 to OptimizerIntegTest.RANDOM_SAMPLES) {
      val initParam = DenseVector.fill[Double](OptimizerIntegTest.PROBLEM_DIMENSION)(r.nextDouble())
      optim.optimize(Array(pt2), new IntegTestObjective(), initParam)
      OptimizerIntegTest.easyOptimizationStatesChecks(optim.getStateTracker.get)
    }
  }

  @Test(dataProvider = "optimizeEasyTestFunction")
  def checkEasyTestFunctionSparkNoInitialValue(optim: Optimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]])
  : Unit = sparkTest("checkEasyTestFunctionSpark") {
    optim.setStateTrackingEnabled(true)
    optim.setMaximumIterations(OptimizerIntegTest.MAX_ITERATIONS)
    optim.setTolerance(OptimizerIntegTest.CONVERGENCE_TOLERANCE)
    val features = new SparseVector[Double](Array(), Array(), OptimizerIntegTest.PROBLEM_DIMENSION)
    val pt = new LabeledPoint(label = 1, features, offset = 0, weight = 1)
    val data = sc.parallelize(Seq(pt))
    optim.optimize(data, new IntegTestObjective())
    OptimizerIntegTest.easyOptimizationStatesChecks(optim.getStateTracker.get)

    //test weighted sample
    val pt2 = new LabeledPoint(label = 1, features, offset = 0, weight = 0.23)
    val data2 = sc.parallelize(Seq(pt2))
    optim.optimize(data2, new IntegTestObjective())
    OptimizerIntegTest.easyOptimizationStatesChecks(optim.getStateTracker.get)
  }

  @Test(dataProvider = "optimizeEasyTestFunction")
  def checkEasyTestFunctionSparkInitialValue(optim: Optimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]])
  : Unit = sparkTest("checkEasyTestFunctionSpark") {
    optim.setStateTrackingEnabled(true)
    optim.setMaximumIterations(OptimizerIntegTest.MAX_ITERATIONS)
    optim.setTolerance(OptimizerIntegTest.CONVERGENCE_TOLERANCE)
    val features = new SparseVector[Double](Array(), Array(), OptimizerIntegTest.PROBLEM_DIMENSION)
    var pt = new LabeledPoint(label = 1, features, offset = 0, weight = 1)
    var data = sc.parallelize(Seq(pt))
    var r = new Random(OptimizerIntegTest.RANDOM_SEED)
    for (iter <- 0 until OptimizerIntegTest.RANDOM_SAMPLES) {
      val initParam = DenseVector.fill[Double](OptimizerIntegTest.PROBLEM_DIMENSION)(r.nextDouble())
      optim.optimize(data, new IntegTestObjective(), initParam)
      assertTrue(optim.stateTrackingEnabled)
      assertTrue(optim.getStateTracker.isDefined)
      assertTrue(optim.isDone)
      OptimizerIntegTest.easyOptimizationStatesChecks(optim.getStateTracker.get)
    }

    //test weighted sample
    pt = new LabeledPoint(label = 1, features, offset = 0, weight = 0)
    data = sc.parallelize(Seq(pt))
    r = new Random(OptimizerIntegTest.RANDOM_SEED)
    for (iter <- 0 until OptimizerIntegTest.RANDOM_SAMPLES) {
      val initParam = DenseVector.fill[Double](OptimizerIntegTest.PROBLEM_DIMENSION)(r.nextDouble())
      optim.optimize(data, new IntegTestObjective(), initParam)
      OptimizerIntegTest.easyOptimizationStatesChecks(optim.getStateTracker.get)
    }
  }
}

object OptimizerIntegTest extends Logging {
  val PROBLEM_DIMENSION: Int = 10
  val MAX_ITERATIONS: Int = 1000 * PROBLEM_DIMENSION
  val CONVERGENCE_TOLERANCE: Double = 1e-12
  val OBJECTIVE_TOLERANCE: Double = 1e-6
  val GRADIENT_TOLERANCE: Double = 1e-6
  val PARAMETER_TOLERANCE: Double = 1e-4
  val MONOTONICITY_TOLERANCE: Double = 1e-6
  val RANDOM_SEED: Long = 314159265359L
  val RANDOM_SAMPLES: Int = 100

  def checkConvergence(history: OptimizationStatesTracker) {
    var lastValue: Double = Double.MaxValue

    history.getTrackedStates.foreach { state =>
      assertTrue(lastValue >= state.value, "Objective should be monotonically decreasing (current=[" + state.value +
        "], previous=[" + lastValue + "])")
      lastValue = state.value
    }
  }

  /**
    * Common checks for the easy test function:
    * <ul>
    * <li>Did we get the expected parameters?</li>
    * <li>Did we get the expected objective?</li>
    * <li>Did we see monotonic convergence?</li>
    * </ul>
    */
  private def easyOptimizationStatesChecks(optimizerStatesTracker: OptimizationStatesTracker): Unit = {

    logInfo(s"Optimizer state: $optimizerStatesTracker")

    // The optimizer should be converged
    assertTrue(optimizerStatesTracker.converged)
    assertFalse(optimizerStatesTracker.getTrackedTimeHistory.isEmpty)
    assertFalse(optimizerStatesTracker.getTrackedStates.isEmpty)
    assertEquals(optimizerStatesTracker.getTrackedStates.length, optimizerStatesTracker.getTrackedTimeHistory.length)

    val optimizedObj = optimizerStatesTracker.getTrackedStates.last.value
    val optimizedGradientNorm = norm(optimizerStatesTracker.getTrackedStates.last.gradient, 2)
    val optimizedParam = optimizerStatesTracker.getTrackedStates.last.coefficients

    if (optimizerStatesTracker.convergenceReason == Some(FunctionValuesConverged)) {
      // Expected answer in terms of optimal objective
      assertEquals(optimizedObj, 0, OBJECTIVE_TOLERANCE, "Optimized objective should be very close to zero (eps=[" +
        OBJECTIVE_TOLERANCE + "])")
    } else if (optimizerStatesTracker.convergenceReason == Some(GradientConverged)) {
      // Expected answer in terms of optimal gradient
      assertEquals(optimizedGradientNorm, 0, GRADIENT_TOLERANCE, "Optimized gradient norm should be very close to " +
        "zero (eps=[" + GRADIENT_TOLERANCE + "])")
    }

    // Expected answer in terms of optimal parameters
    optimizedParam.foreachPair { (idx, x) =>
      assertEquals(x, TestObjective.CENTROID, PARAMETER_TOLERANCE, "Optimized parameter for index [" + idx +
        "] should be close to TestObjective.CENTROID (eps=[" + PARAMETER_TOLERANCE + "]")
    }

    // Monotonic convergence
    checkConvergence(optimizerStatesTracker)
  }
}
