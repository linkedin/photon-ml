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

import scala.math.abs

import breeze.linalg.{Vector, sum}
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.function.svm.SingleNodeSmoothedHingeLossFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.test.CommonTestUtils

/**
 * Test the base function in GeneralizedLinearOptimizationProblem common to all optimization problems.
 */
class GeneralizedLinearOptimizationProblemTest {
  import CommonTestUtils._
  import GeneralizedLinearOptimizationProblemTest._

  @Test
  def testInitializeZeroModel(): Unit = {
    val optimizer = mock(classOf[Optimizer[ObjectiveFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val objective = mock(classOf[SingleNodeSmoothedHingeLossFunction])
    val regularization = NoRegularizationContext

    doReturn(Some(statesTracker)).when(optimizer).getStateTracker

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
    val hingeProblem = new MockOptimizationProblem(
      optimizer,
      objective,
      SmoothedHingeLossLinearSVMModel.apply,
      regularization)

    val logisticModel = logisticProblem.publicInitializeZeroModel(DIMENSION)
    val linearModel = linearProblem.publicInitializeZeroModel(DIMENSION)
    val poissonModel = poissonProblem.publicInitializeZeroModel(DIMENSION)
    val hingeModel = hingeProblem.publicInitializeZeroModel(DIMENSION)

    assertTrue(logisticModel.isInstanceOf[LogisticRegressionModel])
    assertEquals(logisticModel.coefficients.means.length, DIMENSION)

    assertTrue(linearModel.isInstanceOf[LinearRegressionModel])
    assertEquals(linearModel.coefficients.means.length, DIMENSION)

    assertTrue(poissonModel.isInstanceOf[PoissonRegressionModel])
    assertEquals(poissonModel.coefficients.means.length, DIMENSION)

    assertTrue(hingeModel.isInstanceOf[SmoothedHingeLossLinearSVMModel])
    assertEquals(hingeModel.coefficients.means.length, DIMENSION)
  }

  @Test
  def testCreateModel(): Unit = {
    val optimizer = mock(classOf[Optimizer[ObjectiveFunction]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val objective = mock(classOf[SingleNodeSmoothedHingeLossFunction])
    val regularization = NoRegularizationContext

    doReturn(Some(statesTracker)).when(optimizer).getStateTracker

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
    val hingeProblem = new MockOptimizationProblem(
      optimizer,
      objective,
      SmoothedHingeLossLinearSVMModel.apply,
      regularization)
    val coefficients = generateDenseVector(DIMENSION)

    val logisticModel = logisticProblem.publicCreateModel(coefficients, None)
    val linearModel = linearProblem.publicCreateModel(coefficients, None)
    val poissonModel = poissonProblem.publicCreateModel(coefficients, None)
    val hingeModel = hingeProblem.publicCreateModel(coefficients, None)

    assertTrue(logisticModel.isInstanceOf[LogisticRegressionModel])
    assertEquals(coefficients, logisticModel.coefficients.means)

    assertTrue(linearModel.isInstanceOf[LinearRegressionModel])
    assertEquals(coefficients, linearModel.coefficients.means)

    assertTrue(poissonModel.isInstanceOf[PoissonRegressionModel])
    assertEquals(coefficients, poissonModel.coefficients.means)

    assertTrue(hingeModel.isInstanceOf[SmoothedHingeLossLinearSVMModel])
    assertEquals(coefficients, hingeModel.coefficients.means)
  }

  @Test
  def testGetRegularizationTermValue(): Unit = {
    val coefficients = new Coefficients(generateDenseVector(DIMENSION))
    val regWeight = 10D
    val alpha = 0.25
    val l1RegWeight = alpha * regWeight
    val l2RegWeight = (1 - alpha) * regWeight
    val expectedL1Term = sum(coefficients.means.map(abs)) * l1RegWeight
    val expectedL2Term = coefficients.means.dot(coefficients.means) * l2RegWeight / 2.0
    val expectedElasticNetTerm = expectedL1Term + expectedL2Term

    val optimizerNoReg = mock(classOf[LBFGS])
    val optimizerL1Reg = mock(classOf[OWLQN])
    val objectiveNoReg = mock(classOf[SingleNodeSmoothedHingeLossFunction])
    val objectiveL2Reg = mock(classOf[L2LossFunction])
    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val initialModel = mock(classOf[GeneralizedLinearModel])

    doReturn(Some(statesTracker)).when(optimizerNoReg).getStateTracker
    doReturn(Some(statesTracker)).when(optimizerL1Reg).getStateTracker

    val problemNone = new MockOptimizationProblem(
      optimizerNoReg,
      objectiveNoReg,
      LogisticRegressionModel.apply,
      NoRegularizationContext)
    val problemL1 = new MockOptimizationProblem(
      optimizerL1Reg,
      objectiveNoReg,
      LogisticRegressionModel.apply,
      L1RegularizationContext)
    val problemL2 = new MockOptimizationProblem(
      optimizerNoReg,
      objectiveL2Reg,
      LogisticRegressionModel.apply,
      L2RegularizationContext)
    val problemElasticNet = new MockOptimizationProblem(
      optimizerL1Reg,
      objectiveL2Reg,
      LogisticRegressionModel.apply,
      ElasticNetRegularizationContext(alpha))

    doReturn(l1RegWeight).when(optimizerL1Reg).l1RegularizationWeight
    doReturn(l2RegWeight).when(objectiveL2Reg).l2RegularizationWeight
    doReturn(coefficients).when(initialModel).coefficients

    assertEquals(0.0, problemNone.getRegularizationTermValue(initialModel), HIGH_PRECISION_TOLERANCE_THRESHOLD)
    assertEquals(expectedL1Term, problemL1.getRegularizationTermValue(initialModel), HIGH_PRECISION_TOLERANCE_THRESHOLD)
    assertEquals(expectedL2Term, problemL2.getRegularizationTermValue(initialModel), HIGH_PRECISION_TOLERANCE_THRESHOLD)
    assertEquals(
      expectedElasticNetTerm,
      problemElasticNet.getRegularizationTermValue(initialModel),
      HIGH_PRECISION_TOLERANCE_THRESHOLD)
  }
}

object GeneralizedLinearOptimizationProblemTest {

  private val DIMENSION = 10

  private class MockOptimizationProblem(
      optimizer: Optimizer[SingleNodeSmoothedHingeLossFunction],
      objectiveFunction: SingleNodeSmoothedHingeLossFunction,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      regularizationContext: RegularizationContext)
    extends GeneralizedLinearOptimizationProblem[SingleNodeSmoothedHingeLossFunction](
      optimizer,
      objectiveFunction,
      glmConstructor,
      false) {

    private val mockGLM = mock(classOf[GeneralizedLinearModel])

    // Public versions of protected methods for testing
    def publicInitializeZeroModel(dimension: Int): GeneralizedLinearModel = initializeZeroModel(dimension)

    def publicCreateModel(coefficients: Vector[Double], variances: Option[Vector[Double]]): GeneralizedLinearModel =
      createModel(coefficients, variances)

    // Override abstract methods for testing
    override def computeVariances(input: Iterable[LabeledPoint], coefficients: Vector[Double]): Option[Vector[Double]] =
      None

    override def run(input: Iterable[LabeledPoint]): GeneralizedLinearModel = mockGLM

    override def run(input: Iterable[LabeledPoint], initialModel: GeneralizedLinearModel): GeneralizedLinearModel =
      mockGLM
  }

  // No way to pass Mixin class type to Mockito, need to define a concrete class
  private class L2LossFunction extends SingleNodeSmoothedHingeLossFunction with L2RegularizationDiff
}
