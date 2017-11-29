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

import org.apache.spark.broadcast.Broadcast
import org.mockito.Mockito._
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.test.{Assertions, CommonTestUtils}

/**
 * Test that LBFGS can shrink the coefficients to zero.
 *
 * The objective function has known minimum when all coefficients are at 4 (see TestObjective). We are only interested
 * in the behavior around the minimum, where:
 *
 *    x_i > 0 for all i
 *
 * which has an obvious analytic solution. This test is based on the above function and verifies the shrinkage of x.
 */
class LBFGSTest {

  @Test
  def testLBFGS(): Unit = {

    val normalizationContext = NoNormalization()
    val normalizationContextBroadcast = mock(classOf[Broadcast[NormalizationContext]])
    doReturn(normalizationContext).when(normalizationContextBroadcast).value

    val lbfgs = new LBFGS(normalizationContext = normalizationContextBroadcast)
    val objective = new TestObjective
    val trainingData = Array(LabeledPoint(0.0, CommonTestUtils.generateDenseVector(1), 0.0, 0.0))
    val initialCoefficients = CommonTestUtils.generateDenseVector(1)
    val (actualCoef, actualValue) = lbfgs.optimize(objective, initialCoefficients)(trainingData)

    Assertions.assertIterableEqualsWithTolerance(actualCoef.toArray, Array(TestObjective.CENTROID), LBFGSTest.EPSILON)
    assertEquals(actualValue, LBFGSTest.EXPECTED_LOSS, LBFGSTest.EPSILON)
  }
}

object LBFGSTest {
  private val EPSILON = 1.0E-6
  private val EXPECTED_LOSS = 0
}
