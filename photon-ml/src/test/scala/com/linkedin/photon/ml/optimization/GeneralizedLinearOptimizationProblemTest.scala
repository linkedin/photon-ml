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
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.sampler.DownSampler
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}
import com.linkedin.photon.ml.test.CommonTestUtils

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.Test

import scala.collection.mutable

class GeneralizedLinearOptimizationProblemTest {
  import GeneralizedLinearOptimizationProblemTest._
  import CommonTestUtils._

  @Test
  def testRun(): Unit = {
    val (problem, optimizer, objectiveFunction) = createProblem

    val dim = 10
    val data = mock(classOf[RDD[LabeledPoint]])
    val initialModel = LogisticRegressionOptimizationProblem.initializeZeroModel(dim)
    val normalizationContext = mock(classOf[NormalizationContext])

    val coefficients = new Coefficients(generateDenseVector(dim))

    doReturn(coefficients.means).when(normalizationContext).transformModelCoefficients(coefficients.means)
    doReturn((coefficients.means, None))
      .when(optimizer)
      .optimize(Matchers.eq(data), Matchers.eq(objectiveFunction), Matchers.any(classOf[Vector[Double]]))

    val statesTracker = mock(classOf[OptimizationStatesTracker])
    val state = OptimizerState(coefficients.means, 0, generateDenseVector(dim), 0)

    doReturn(Array(state)).when(statesTracker).getTrackedStates
    doReturn(Some(statesTracker)).when(optimizer).getStateTracker

    val model = problem.run(data, initialModel, normalizationContext)

    assertEquals(coefficients, model.coefficients)
    assertEquals(problem.getModelTracker.get.length, 1)
  }
}

object GeneralizedLinearOptimizationProblemTest {
  def createProblem() = {
    val optimizer = mock(classOf[Optimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]]])
    val sampler = mock(classOf[DownSampler])
    val objectiveFunction = mock(classOf[TwiceDiffFunction[LabeledPoint]])
    val regularizationContext = mock(classOf[RegularizationContext])
    val regularizationWeight = 0D
    val modelTrackerBuilder = Some(new mutable.ListBuffer[ModelTracker]())
    val treeAggregateDepth = 1
    val isComputingVariances: Boolean = false

    val problem = new LogisticRegressionOptimizationProblem(
      optimizer, sampler, objectiveFunction, regularizationContext, regularizationWeight, modelTrackerBuilder,
      treeAggregateDepth, isComputingVariances)

    (problem, optimizer, objectiveFunction)
  }
}
