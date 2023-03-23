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
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.test.CommonTestUtils.generateDenseVector

/**
 * Test the base function in [[GeneralizedLinearOptimizationProblem]] common to all optimization problems.
 */
class GeneralizedLinearOptimizationProblemTest {

  import GeneralizedLinearOptimizationProblemTest._

  @Test
  def testInitializeZeroModel(): Unit = {
    val optimizer = mock(classOf[Optimizer[ObjectiveFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val objective = mock(classOf[SingleNodeObjectiveFunction])
    val regularization = NoRegularizationContext

    doReturn(statesTracker).when(optimizer).getStateTracker

    val logisticProblem = new MockOptimizationProblem(
      optimizer,
      objective,
      LogisticRegressionModel.apply,
      regularization)
    val linearProblem = new MockOptimizationProblem(
      optimizer,
      objective,
      LinearRegressionModel.apply,
      regularization)
    val poissonProblem = new MockOptimizationProblem(
      optimizer,
      objective,
      PoissonRegressionModel.apply,
      regularization)

    val logisticModel = logisticProblem.publicInitializeZeroModel(DIMENSION)
    val linearModel = linearProblem.publicInitializeZeroModel(DIMENSION)
    val poissonModel = poissonProblem.publicInitializeZeroModel(DIMENSION)

    assertTrue(logisticModel.isInstanceOf[LogisticRegressionModel])
    assertEquals(logisticModel.coefficients.means.length, DIMENSION)

    assertTrue(linearModel.isInstanceOf[LinearRegressionModel])
    assertEquals(linearModel.coefficients.means.length, DIMENSION)

    assertTrue(poissonModel.isInstanceOf[PoissonRegressionModel])
    assertEquals(poissonModel.coefficients.means.length, DIMENSION)
  }

  @Test
  def testCreateModel(): Unit = {

    val optimizer = mock(classOf[Optimizer[ObjectiveFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val objective = mock(classOf[SingleNodeObjectiveFunction])
    val regularization = NoRegularizationContext

    doReturn(statesTracker).when(optimizer).getStateTracker

    val logisticProblem = new MockOptimizationProblem(
      optimizer,
      objective,
      LogisticRegressionModel.apply,
      regularization)
    val linearProblem = new MockOptimizationProblem(
      optimizer,
      objective,
      LinearRegressionModel.apply,
      regularization)
    val poissonProblem = new MockOptimizationProblem(
      optimizer,
      objective,
      PoissonRegressionModel.apply,
      regularization)
    val coefficients = generateDenseVector(DIMENSION)

    val logisticModel = logisticProblem.publicCreateModel(coefficients, None)
    val linearModel = linearProblem.publicCreateModel(coefficients, None)
    val poissonModel = poissonProblem.publicCreateModel(coefficients, None)

    assertTrue(logisticModel.isInstanceOf[LogisticRegressionModel])
    assertEquals(coefficients, logisticModel.coefficients.means)

    assertTrue(linearModel.isInstanceOf[LinearRegressionModel])
    assertEquals(coefficients, linearModel.coefficients.means)

    assertTrue(poissonModel.isInstanceOf[PoissonRegressionModel])
    assertEquals(coefficients, poissonModel.coefficients.means)
  }
}

object GeneralizedLinearOptimizationProblemTest {

  private val DIMENSION = 10

  private class MockOptimizationProblem(
      optimizer: Optimizer[SingleNodeObjectiveFunction],
      objectiveFunction: SingleNodeObjectiveFunction,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      regularizationContext: RegularizationContext)
    extends GeneralizedLinearOptimizationProblem[SingleNodeObjectiveFunction](
      optimizer,
      objectiveFunction,
      glmConstructor,
      VarianceComputationType.NONE) {

    private val mockGLM = mock(classOf[GeneralizedLinearModel])
    private val mockStateTracker = mock(classOf[OptimizationStatesTracker])

    //
    // Public versions of protected methods for testing
    //

    /**
     * Public version of [[initializeZeroModel]].
     */
    def publicInitializeZeroModel(dimension: Int): GeneralizedLinearModel = initializeZeroModel(dimension)

    /**
     * Publi version of [[createModel]]
     */
    def publicCreateModel(coefficients: Vector[Double], variances: Option[Vector[Double]]): GeneralizedLinearModel =
      createModel(coefficients, variances)

    //
    // Override abstract methods for testing
    //

    /**
     * Unused - needs definition for testing.
     */
    override def computeVariances(input: Iterable[LabeledPoint], coefficients: Vector[Double]): Option[Vector[Double]] =
      None

    /**
     * Unused - needs definition for testing.
     */
    override def run(input: Iterable[LabeledPoint]): (GeneralizedLinearModel, OptimizationStatesTracker) = (mockGLM, mockStateTracker)

    /**
     * Unused - needs definition for testing.
     */
    override def run(input: Iterable[LabeledPoint], initialModel: GeneralizedLinearModel): (GeneralizedLinearModel, OptimizationStatesTracker) =
      (mockGLM, mockStateTracker)
  }
}
