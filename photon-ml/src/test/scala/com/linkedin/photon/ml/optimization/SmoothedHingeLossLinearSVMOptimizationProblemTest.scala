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

import org.apache.spark.rdd.RDD
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.{LabeledPoint, SimpleObjectProvider}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.test.CommonTestUtils

class SmoothedHingeLossLinearSVMOptimizationProblemTest {
  import CommonTestUtils._
  import SmoothedHingeLossLinearSVMOptimizationProblemTest._

  @Test
  def testUpdateObjective(): Unit = {
    val problem = createProblem()
    val normalizationContext = new SimpleObjectProvider(mock(classOf[NormalizationContext]))
    val regularizationWeight = 1D

    assertNotEquals(problem.regularizationWeight, regularizationWeight)

    val updatedProblem = problem.updateObjective(normalizationContext, regularizationWeight)

    assertEquals(updatedProblem.regularizationWeight, regularizationWeight)
  }

  @Test
  def testInitializeZeroModel(): Unit = {
    val problem = createProblem()
    val zeroModel = problem.initializeZeroModel(DIMENSIONS)

    assertEquals(zeroModel.coefficients, Coefficients.initializeZeroCoefficients(DIMENSIONS))
  }

  @Test
  def testCreateModel(): Unit = {
    val problem = createProblem()
    val coefficients = generateDenseVector(DIMENSIONS)
    val model = problem.createModel(coefficients, None)

    assertEquals(model.coefficients.means, coefficients)
  }

  @Test
  def testComputeVariancesDisabled(): Unit = {
    val problem = createProblem()
    val input = mock(classOf[RDD[LabeledPoint]])
    val coefficients = generateDenseVector(DIMENSIONS)

    assertEquals(problem.computeVariances(input, coefficients), None)
  }

  @Test
  def testComputeVariancesEnabled(): Unit = {
    val problem = createProblem(computeVariance = true)
    val input = mock(classOf[RDD[LabeledPoint]])
    val coefficients = generateDenseVector(DIMENSIONS)

    assertEquals(problem.computeVariances(input, coefficients), None)
  }
}

object SmoothedHingeLossLinearSVMOptimizationProblemTest {
  val DIMENSIONS = 10

  def createProblem(computeVariance: Boolean = false) = {
    val config = new GLMOptimizationConfiguration(
      optimizerConfig = OptimizerConfig(OptimizerType.LBFGS, 100, 1E-10, None))
    val treeAggregateDepth = 1
    val isTrackingState = false

    SmoothedHingeLossLinearSVMOptimizationProblem.buildOptimizationProblem(
      config,
      treeAggregateDepth,
      isTrackingState,
      computeVariance)
  }
}
