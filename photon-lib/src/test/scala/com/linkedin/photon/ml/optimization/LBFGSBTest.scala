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

import breeze.linalg.DenseVector
import org.apache.spark.broadcast.Broadcast
import org.mockito.Mockito.{doReturn, mock}
import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.test.{Assertions, CommonTestUtils}

class LBFGSBTest {

  @DataProvider(name = "dataProvider")
  def dataProvider(): Array[Array[Any]] = {
    Array(
      //Cannot use Double.PositiveInfinity or Double.NegativeInfinity because of mean used later
      Array(Array(-10.0), Array(10.0), Array(4.0), 0.0),
      Array(Array(-5.0), Array(5.0), Array(4.0), 0.0),
      Array(Array(-10.0), Array(3.0), Array(3.0), 1.0),
      Array(Array(5.0), Array(10.0), Array(5.0), 1.0)
    )
  }

  @Test(dataProvider = "dataProvider")
  def testLBFGSB(
      lowerBounds: Array[Double],
      upperBounds: Array[Double],
      expectedCoef: Array[Double],
      expectedValue: Double): Unit = {

    val normalizationContext = NoNormalization()
    val normalizationContextBroadcast = mock(classOf[Broadcast[NormalizationContext]])
    doReturn(normalizationContext).when(normalizationContextBroadcast).value

    val lbfgsb = new LBFGSB(DenseVector(lowerBounds), DenseVector(upperBounds), normalizationContext = normalizationContextBroadcast)
    val objective = new TestObjective
    val trainingData = Array(LabeledPoint(0.0, CommonTestUtils.generateDenseVector(expectedCoef.length), 0.0, 0.0))
    // update each initial coefficient in the range [lowerBounds, upperBounds]
    val initialCoefficients = DenseVector.zeros[Double](expectedCoef.length)
    for(i <- 0 until initialCoefficients.length){
      initialCoefficients(i) = CommonTestUtils.generateDenseVector(1, (lowerBounds(i) + upperBounds(i))/2)(0)
    }
    val (actualCoef, actualValue) = lbfgsb.optimize(objective, initialCoefficients)(trainingData)

    Assertions.assertIterableEqualsWithTolerance(actualCoef.toArray, expectedCoef, LBFGSBTest.EPSILON)
    assertEquals(actualValue, expectedValue, LBFGSBTest.EPSILON)
  }
}

object LBFGSBTest {
  private val EPSILON = 1.0E-6
}
