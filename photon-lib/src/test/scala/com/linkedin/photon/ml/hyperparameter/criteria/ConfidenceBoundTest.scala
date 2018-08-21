/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.hyperparameter.criteria

import breeze.linalg.DenseVector
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance

/**
 * Test cases for the ConfidenceBound class
 */
class ConfidenceBoundTest {

  val TOL = 1e-3

  /**
   * Test data
   */
  @DataProvider
  def modelDataProvider() =
    Array(
      Array(DenseVector(1.0, 2.0, 3.0), DenseVector(1.0, 2.0, 3.0), 0),
      Array(DenseVector(-4.0, 5.0, -6.0), DenseVector(3.0, 2.0, 1.0), 1))

  /**
   * Unit tests for [[ConfidenceBound.apply]]
   */
  @Test(dataProvider = "modelDataProvider")
  def testApply(mu: DenseVector[Double], sigma: DenseVector[Double], testSetIndex: Int): Unit = {

    val confidenceBound = new ConfidenceBound
    val predicted = confidenceBound(mu, sigma)

    val expected = testSetIndex match {
      case 0 => DenseVector(-1.0000, -0.8284, -0.4641)
      case 1 => DenseVector(-7.4641, 2.1716, -8.0000)
    }

    assertIterableEqualsWithTolerance(predicted.toArray, expected.toArray, TOL)
  }
}
