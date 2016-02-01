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
package com.linkedin.photon.ml.normalization

import breeze.linalg.{DenseVector, Vector}
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import scala.util.Random


/**
 * Test [[NormalizationContext]]
 * @author dpeng
 */
class NormalizationContextTest {
  val dim = 5
  val seed = 1L
  val random = new Random(seed)
  val sigma = 5.0
  val testCoef = DenseVector(1.0, -1.0, 0.0, 0.5, -0.5)
  val testFactor = DenseVector(0.5, 2.0, -0.1, -0.4, 1.0)
  val testShift = DenseVector(-0.5, 0.1, 4.0, -2.0, 0.0)
  val epsilon = 1.0E-9

  /**
   * Verify model coefficients and feature transformation is compatible. The margin in the original scale and the
   * transformed scale should be the same.
   * @param factors
   * @param shifts
   * @param addIntercept
   */
  @Test(dataProvider = "dataProvider")
  def testTransformationWithIntercept(factors: Option[Vector[Double]], shifts: Option[Vector[Double]],
                                      addIntercept: Boolean): Unit = {

    val vector = getRandomVector(addIntercept)
    val normalizationContext = NormalizationContext(factors, shifts, Some(dim - 1))
    val transformedVector = normalizationContext.transformVector(vector)
    val originalCoefs = normalizationContext.transformModelCoefficients(testCoef)
    val expectedMargin = originalCoefs.dot(vector)
    val actual = testCoef.dot(transformedVector)
    Assert.assertEquals(actual, expectedMargin, epsilon)
  }

  @DataProvider
  def dataProvider(): Array[Array[Any]] = {
    Array(
      Array(None, None, true),
      Array(Some(testFactor), None, true),
      Array(None, Some(testShift), true),
      Array(Some(testFactor), Some(testShift), true),
      Array(None, None, false),
      Array(Some(testFactor), None, false)
    )
  }

  private def getRandomVector(addIntercept: Boolean ): Vector[Double] = {
    if (addIntercept) {
      val ar = for (i <- 0 until dim - 1) yield random.nextGaussian() * sigma
      new DenseVector(ar.toArray :+ 1.0)
    } else {
      val ar = for (i <- 0 until dim) yield random.nextGaussian() * sigma
      new DenseVector(ar.toArray)
    }
  }

  @Test(dataProvider = "invalidInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidNormalizationContext(factors: Option[Vector[Double]], shifts: Option[Vector[Double]],
                                      addIntercept: Boolean): Unit = {
    new NormalizationContext(factors, shifts, None)
  }

  @DataProvider
  def invalidInput(): Array[Array[Any]] = {
    Array(
      Array(None, Some(testShift), false),
      Array(Some(testFactor), Some(testShift), false),
      Array(Some(testFactor), Some(testShift :+ 1.0), true),
      Array(Some(testFactor), Some(testShift :+ 1.0), false)
    )
  }
}
