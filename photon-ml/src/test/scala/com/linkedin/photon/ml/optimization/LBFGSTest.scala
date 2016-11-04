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

import java.util.Random

import breeze.linalg.DenseVector
import org.apache.spark.broadcast.Broadcast
import org.mockito.Mockito._
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.{NormalizationContext, NoNormalization}
import com.linkedin.photon.ml.test.Assertions
import com.linkedin.photon.ml.test.CommonTestUtils

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
  import CommonTestUtils._

  val random = new Random(LBFGSTest.RANDOM_SEED)

  private def getRandomInput(dim: Int): DenseVector[Double] = {
    DenseVector(Seq.fill(dim)(random.nextGaussian).toArray)
  }

  @DataProvider(name = "dataProvider")
  def dataProvider(): Array[Array[Any]] = {
    Array(
      // Note that expected value here is the value of the function at the unconstrained optima since the
      // projection happens after it
      Array(Some(Map[Int, (Double, Double)]()), Array(4.0)),
      Array(Some(Map[Int, (Double, Double)](0 -> (3.0, 5.0))), Array(4.0)),
      Array(Some(Map[Int, (Double, Double)](0 -> (Double.NegativeInfinity, 3.0))), Array(3.0)),
      Array(Some(Map[Int, (Double, Double)](0 -> (5.0, Double.PositiveInfinity))), Array(5.0))
    )
  }

  @Test(dataProvider = "dataProvider")
  def testLBFGS(
      constraintMap: Option[Map[Int, (Double, Double)]],
      expectedCoef: Array[Double]): Unit = {

    val normalizationContext = NoNormalization()
    val normalizationContextBroadcast = mock(classOf[Broadcast[NormalizationContext]])
    doReturn(normalizationContext).when(normalizationContextBroadcast).value

    val lbfgs = new LBFGS(constraintMap = constraintMap, normalizationContext = normalizationContextBroadcast)
    val objective = new TestObjective
    val trainingData = Array(LabeledPoint(0.0, generateDenseVector(expectedCoef.length), 0.0, 0.0))
    val initialCoefficients = generateDenseVector(expectedCoef.length)
    val (actualCoef, actualValue) = lbfgs.optimize(objective, initialCoefficients)(trainingData)

    Assertions.assertIterableEqualsWithTolerance(actualCoef.toArray, expectedCoef, LBFGSTest.EPSILON)
    Assert.assertEquals(actualValue, LBFGSTest.EXPECTED_LOSS, LBFGSTest.EPSILON)
  }
}

object LBFGSTest {
  val EPSILON = 1.0E-6
  val RANDOM_SEED = 1
  val EXPECTED_LOSS = 0
}
