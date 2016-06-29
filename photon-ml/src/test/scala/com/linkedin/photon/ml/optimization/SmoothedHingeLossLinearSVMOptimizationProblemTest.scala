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

import com.linkedin.photon.ml.data.{LabeledPoint, SimpleObjectProvider}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.test.CommonTestUtils

import org.apache.spark.rdd.RDD
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.Test

class SmoothedHingeLossLinearSVMOptimizationProblemTest {
  import SmoothedHingeLossLinearSVMOptimizationProblemTest._
  import CommonTestUtils._

  @Test
  def testUpdateObjective(): Unit = {
    val problem = createProblem
    val normalizationContext = new SimpleObjectProvider(mock(classOf[NormalizationContext]))
    val regularizationWeight = 1D

    assertNotEquals(problem.regularizationWeight, regularizationWeight)

    val updatedProblem = problem.updateObjective(normalizationContext, regularizationWeight)

    assertEquals(updatedProblem.regularizationWeight, regularizationWeight)
  }

  @Test
  def testInitializeZeroModel(): Unit = {
    val problem = createProblem
    val zeroModel = problem.initializeZeroModel(Dimensions)

    assertEquals(zeroModel.coefficients, Coefficients.initializeZeroCoefficients(Dimensions))
  }

  @Test
  def testCreateModel(): Unit = {
    val problem = createProblem
    val coefficients = generateDenseVector(Dimensions)
    val model = problem.createModel(coefficients, None)

    assertEquals(model.coefficients.means, coefficients)
  }

  @Test
  def testComputeVariances(): Unit = {
    val problem = createProblem
    val input = mock(classOf[RDD[LabeledPoint]])
    val coefficients = generateDenseVector(Dimensions)

    // TODO: computeVarainces is currently disabled. This test will need to be updated when the default changes
    assertEquals(problem.computeVariances(input, coefficients), None)
  }
}

object SmoothedHingeLossLinearSVMOptimizationProblemTest {
  val Dimensions = 10

  def createProblem() = {
    val optimizerConfig = OptimizerConfig(OptimizerType.LBFGS, 1, 1e-5, None)
    val config = new GLMOptimizationConfiguration(optimizerConfig)
    val treeAggregateDepth = 1
    val isTrackingState = false

    SmoothedHingeLossLinearSVMOptimizationProblem.buildOptimizationProblem(
      config, treeAggregateDepth, isTrackingState)
  }
}
