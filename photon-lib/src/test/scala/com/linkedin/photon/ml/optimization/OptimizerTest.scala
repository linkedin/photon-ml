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

import java.util.Random

import breeze.linalg.{DenseVector, SparseVector, Vector, norm}
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.util._

/**
 * Verify that core optimizers do reasonable things on small test problems.
 */
//class OptimizerTest extends SparkTestUtils with Logging {
class OptimizerTest extends Logging {
  import OptimizerTest._

  @DataProvider(parallel = true)
  def optimizersUsingInitialValue(): Array[Array[Object]] = {
    Array(
      Array(new LBFGS(
        tolerance = CONVERGENCE_TOLERANCE,
        maxNumIterations = MAX_ITERATIONS,
        normalizationContext = NORMALIZATION_MOCK)),
      Array(new TRON(
        tolerance = CONVERGENCE_TOLERANCE,
        maxNumIterations = MAX_ITERATIONS,
        normalizationContext = NORMALIZATION_MOCK)))
  }

  @DataProvider(parallel = true)
  def optimizersNotUsingInitialValue(): Array[Array[Object]] = {
    Array(
      Array(new LBFGS(
        tolerance = CONVERGENCE_TOLERANCE,
        maxNumIterations = MAX_ITERATIONS,
        normalizationContext = NORMALIZATION_MOCK)),
      Array(new TRON(
        tolerance = CONVERGENCE_TOLERANCE,
        maxNumIterations = MAX_ITERATIONS,
        normalizationContext = NORMALIZATION_MOCK)))
  }

  // TODO: Currently the test objective function used by this test ignores weights, so testing points with varying
  //       weights is pointless

  @Test(dataProvider = "optimizersUsingInitialValue")
  def checkEasyTestFunctionLocalNoInitialValue(optimizer: Optimizer[TwiceDiffFunction]) : Unit = {
    val features = new SparseVector[Double](Array(), Array(), PROBLEM_DIMENSION)
    val objective = new TestObjective

    // Test unweighted sample
    val pt1 = Seq(new LabeledPoint(label = 1, features, offset = 0, weight = 1))
    val zero = Vector.zeros[Double](PROBLEM_DIMENSION)
    optimizer.optimize(objective, zero)(pt1)
    easyOptimizationStatesChecks(optimizer.getStateTracker)

    // Test weighted sample
    val pt2 = new LabeledPoint(label = 1, features, offset = 0, weight = 1.5)
    optimizer.optimize(objective, zero)(Seq(pt2))
    easyOptimizationStatesChecks(optimizer.getStateTracker)
  }

  @Test(dataProvider = "optimizersNotUsingInitialValue")
  def checkEasyTestFunctionLocalInitialValue(optimizer: Optimizer[TwiceDiffFunction]): Unit = {
    val features = new SparseVector[Double](Array(), Array(), PROBLEM_DIMENSION)
    val pt = new LabeledPoint(label = 1, features, offset = 0, weight = 1)
    val pt2 = new LabeledPoint(label = 1, features, offset = 0, weight = 0.5)
    val r = new Random(RANDOM_SEED)
    val objective = new TestObjective

    // Test unweighted sample
    for (_ <- 0 to RANDOM_SAMPLES) {
      val initParam = DenseVector.fill[Double](PROBLEM_DIMENSION)(r.nextDouble())
      optimizer.optimize(objective, initParam)(Seq(pt))

      assertTrue(optimizer.isDone)
      easyOptimizationStatesChecks(optimizer.getStateTracker)
    }

    // Test weighted sample
    for (_ <- 0 to RANDOM_SAMPLES) {
      val initParam = DenseVector.fill[Double](PROBLEM_DIMENSION)(r.nextDouble())
      optimizer.optimize(objective, initParam)(Seq(pt2))

      easyOptimizationStatesChecks(optimizer.getStateTracker)
    }
  }
}

object OptimizerTest extends Logging {

  private val PROBLEM_DIMENSION: Int = 5
  private val MAX_ITERATIONS: Int = 1000 * PROBLEM_DIMENSION
  private val CONVERGENCE_TOLERANCE: Double = 1e-12
  private val OBJECTIVE_TOLERANCE: Double = 1e-6
  private val GRADIENT_TOLERANCE: Double = 1e-6
  private val PARAMETER_TOLERANCE: Double = 1e-4
  private val RANDOM_SEED: Long = 314159265359L
  private val RANDOM_SAMPLES: Int = 100
  private val NORMALIZATION = NoNormalization()
  private val NORMALIZATION_MOCK = mock(classOf[BroadcastWrapper[NormalizationContext]])

  doReturn(NORMALIZATION).when(NORMALIZATION_MOCK).value

  /**
   * Check if optimization ended because either the objective function value or gradient converged.
   *
   * @return True if either the function values or gradient converged, false otherwise
   */
  protected[optimization] def checkConvergence(convergenceReasonOpt: Option[ConvergenceReason]): Unit =
    assertTrue(
      convergenceReasonOpt.exists { convergenceReason =>
        convergenceReason == FunctionValuesConverged || convergenceReason == GradientConverged
      })

  /**
   *
   * @param history
   */
  protected[optimization] def checkMonotonicConvergence(history: OptimizationStatesTracker): Unit = history
    .getTrackedStates
    .foldLeft(Double.MaxValue) { case (lastValue: Double, state: OptimizerState) =>
      assertTrue(
        lastValue >= state.loss,
        s"Objective should be monotonically decreasing (current=[${state.loss}], previous=[$lastValue])")

      state.loss
    }

  /**
   * Common checks for the easy test function:
   * <ul>
   * <li>Did we get the expected parameters?</li>
   * <li>Did we get the expected objective?</li>
   * <li>Did we see monotonic convergence?</li>
   * </ul>
   *
   * @param optimizerStatesTracker
   */
  private def easyOptimizationStatesChecks(optimizerStatesTracker: OptimizationStatesTracker): Unit = {

    logger.info(s"Optimizer state: $optimizerStatesTracker")

    // The optimizer should be converged
    checkConvergence(optimizerStatesTracker.convergenceReason)

    assertFalse(optimizerStatesTracker.getTrackedTimeHistory.isEmpty)
    assertFalse(optimizerStatesTracker.getTrackedStates.isEmpty)
    assertEquals(optimizerStatesTracker.getTrackedStates.length, optimizerStatesTracker.getTrackedTimeHistory.length)

    val optimizedObj = optimizerStatesTracker.getTrackedStates.last.loss
    val optimizedGradientNorm = norm(optimizerStatesTracker.getTrackedStates.last.gradient, 2)
    val optimizedParam = optimizerStatesTracker.getTrackedStates.last.coefficients

    if (optimizerStatesTracker.convergenceReason.forall(_ == FunctionValuesConverged)) {
      // Expected answer in terms of optimal objective
      assertEquals(
        optimizedObj,
        0,
        OBJECTIVE_TOLERANCE,
        s"Optimized objective should be very close to zero (eps=[$OBJECTIVE_TOLERANCE])")
    } else if (optimizerStatesTracker.convergenceReason.forall(_ == GradientConverged)) {
      // Expected answer in terms of optimal gradient
      assertEquals(
        optimizedGradientNorm,
        0,
        GRADIENT_TOLERANCE,
        s"Optimized gradient norm should be very close to zero (eps=[$GRADIENT_TOLERANCE])")
    }

    // Expected answer in terms of optimal parameters
    optimizedParam.foreachPair { (idx, x) =>
      assertEquals(
        x,
        TestObjective.CENTROID,
        PARAMETER_TOLERANCE,
        s"Optimized parameter for index [$idx] should be close to TestObjective.CENTROID (eps=[$PARAMETER_TOLERANCE]")
    }

    // Monotonic convergence
    checkMonotonicConvergence(optimizerStatesTracker)
  }
}
