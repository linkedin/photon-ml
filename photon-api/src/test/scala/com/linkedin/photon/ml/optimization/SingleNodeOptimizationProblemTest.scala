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

import breeze.linalg.Vector
import org.apache.spark.broadcast.Broadcast
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.glm.SingleNodeGLMLossFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.CommonTestUtils.generateDenseVector

/**
 * Test that the [[SingleNodeOptimizationProblem]] runs properly and can correctly skip variance computation if it is
 * disabled. For additional variance computation tests, see the [[DistributedOptimizationProblemIntegTest]].
 */
class SingleNodeOptimizationProblemTest {

  private val DIMENSIONS: Int = 5

  @Test
  def testComputeVariancesDisabled(): Unit = {
    val optimizer = mock(classOf[Optimizer[SingleNodeGLMLossFunction]])
    val objectiveFunction = mock(classOf[SingleNodeGLMLossFunction])
    val statesTracker = mock(classOf[OptimizationStatesTracker])

    doReturn(Some(statesTracker)).when(optimizer).getStateTracker

    val problem = new SingleNodeOptimizationProblem(
      optimizer,
      objectiveFunction,
      LogisticRegressionModel.apply,
      isComputingVariances = false)
    val trainingData = mock(classOf[Iterable[LabeledPoint]])
    val coefficients = mock(classOf[Vector[Double]])

    assertEquals(problem.computeVariances(trainingData, coefficients), None)
  }

  @Test
  def testRun(): Unit = {
    val coefficients = new Coefficients(generateDenseVector(DIMENSIONS))

    val optimizer = mock(classOf[Optimizer[SingleNodeGLMLossFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val objectiveFunction = mock(classOf[SingleNodeGLMLossFunction])
    val initialModel = mock(classOf[GeneralizedLinearModel])
    val normalizationContext = mock(classOf[NormalizationContext])
    val normalizationContextBroadcast = mock(classOf[Broadcast[NormalizationContext]])
    val trainingData = mock(classOf[Iterable[LabeledPoint]])

    doReturn(Some(statesTracker)).when(optimizer).getStateTracker

    val problem = new SingleNodeOptimizationProblem(
      optimizer,
      objectiveFunction,
      LogisticRegressionModel.apply,
      isComputingVariances = false)

    doReturn(normalizationContextBroadcast).when(optimizer).getNormalizationContext
    doReturn(normalizationContext).when(normalizationContextBroadcast).value
    doReturn(coefficients).when(initialModel).coefficients
    doReturn((coefficients.means, None))
      .when(optimizer)
      .optimize(objectiveFunction, coefficients.means)(trainingData)
    val state = OptimizerState(coefficients.means, 0, generateDenseVector(DIMENSIONS), 0)
    doReturn(Array(state)).when(statesTracker).getTrackedStates
    doReturn(coefficients.means).when(normalizationContext).transformModelCoefficients(coefficients.means)

    val model = problem.run(trainingData, initialModel)

    assertEquals(coefficients, model.coefficients)
    assertEquals(problem.getModelTracker.get.length, 1)
  }
}
