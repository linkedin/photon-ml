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

import org.mockito.Mockito._
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.test.{Assertions, CommonTestUtils}
import com.linkedin.photon.ml.util.BroadcastWrapper

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
 *    (x_1 - 4)^2 + (x_2 - 4)^2 + ... + L1Weight * Abs(Sum(x_i))
 *
 * which has obvious analytic solution. This test is based on the above function and verifies the shrinkage of x.
 */
class OWLQNTest {

  @DataProvider(name = "dataProvider")
  def dataProvider(): Array[Array[Any]] = {
    Array(
      Array(1.0, Array(3.5, 3.5), 7.5),
      Array(2.0, Array(3.0, 3.0), 14.0),
      Array(8.0, Array(0.0, 0.0), 32.0)
    )
  }

  @Test(dataProvider = "dataProvider")
  def testOWLQN(
      l1Weight: Double,
      expectedCoef: Array[Double],
      expectedValue: Double): Unit = {

    val normalizationContext = NoNormalization()
    val normalizationContextBroadcast = mock(classOf[BroadcastWrapper[NormalizationContext]])
    doReturn(normalizationContext).when(normalizationContextBroadcast).value


    val owlqn = new OWLQN(
      l1RegWeight = l1Weight,
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
