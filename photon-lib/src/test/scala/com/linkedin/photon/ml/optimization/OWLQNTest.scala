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

import org.apache.spark.broadcast.Broadcast
import org.mockito.Mockito._
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.test.{Assertions, CommonTestUtils}

/**
 * Test that OWLQN can shrink the coefficients to zero.
 *
 * The objective function, prior to regularization, had known minimum when all coefficients are 4 (see TestObjective).
 * We are only interested in the behavior around the minimum, where:
 *
 *    x_i > 0 for all i
 *
 * Thus the function to be optimized becomes:
 *
 *    (x_1 - 4)^2 + (x_2 - 4)^2 + ... + Abs(Sum(x_i))
 *
 * which has obvious analytic solution. This test is based on the above function and verifies the shrinkage of x.
 */
class OWLQNTest {

  @DataProvider(name = "dataProvider")
  def dataProvider(): Array[Array[Any]] = {
    Array(
      Array(1.0, None, Array(3.5, 3.5), 7.5),
      Array(2.0, None, Array(3.0, 3.0), 14.0),
      Array(8.0, None, Array(0.0, 0.0), 32.0),

      // note that expected value here is the value of the function at the unconstrained optima since the
      // projection happens after it
      Array(1.0, Some(Map[Int, (Double, Double)](0 -> (2.0, 3.0))), Array(3.0), 3.75),
      Array(2.0, Some(Map[Int, (Double, Double)](0 -> (-2.0, -1.0))), Array(-1.0), 7),
      Array(8.0, Some(Map[Int, (Double, Double)](0 -> (3.5, Double.PositiveInfinity))), Array(3.5), 16)
    )
  }

  @Test(dataProvider = "dataProvider")
  def testOWLQN(
      l1Weight: Double,
      constraintMap: Option[Map[Int, (Double, Double)]],
      expectedCoef: Array[Double],
      expectedValue: Double): Unit = {

    val normalizationContext = NoNormalization()
    val normalizationContextBroadcast = mock(classOf[Broadcast[NormalizationContext]])
    doReturn(normalizationContext).when(normalizationContextBroadcast).value


    val owlqn = new OWLQN(
      l1RegWeight = l1Weight,
      constraintMap = constraintMap,
      normalizationContext = normalizationContextBroadcast)
    val objective = new TestObjective
    val trainingData = Array(LabeledPoint(0.0, CommonTestUtils.generateDenseVector(expectedCoef.length), 0.0, 0.0))
    val initialCoefficients = CommonTestUtils.generateDenseVector(expectedCoef.length)
    val (actualCoef, actualValue) = owlqn.optimize(objective, initialCoefficients)(trainingData)

    Assertions.assertIterableEqualsWithTolerance(actualCoef.toArray, expectedCoef, OWLQNTest.EPSILON)
    Assert.assertEquals(actualValue, expectedValue, OWLQNTest.EPSILON)
  }
}

object OWLQNTest {
  private val EPSILON = 1.0E-6
}
