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
 * Unit tests for [[ExpectedImprovement]].
 */
class ExpectedImprovementTest {

  val TOL = 1e-3

  /**
   * Provide test data.
   */
  @DataProvider
  def modelDataProvider() =
    Array(
      Array(DenseVector(1.0, 2.0, 3.0), DenseVector(1.0, 2.0, 3.0), DenseVector(0.0833, 0.0503, 0.0292)),
      Array(DenseVector(-4.0, 5.0, -6.0), DenseVector(3.0, 2.0, 1.0), DenseVector(4.0062, 0.0000, 6.0000)))

  /**
   * Test that the expected improvement over an evaluation can be correctly predicted for a vector of means and standard
   * deviations.
   *
   * @param mu Vector of means
   * @param sigma Vector of standard deviations
   * @param expectedResult Vector of expected improvements
   */
  @Test(dataProvider = "modelDataProvider")
  def testApply(mu: DenseVector[Double], sigma: DenseVector[Double], expectedResult: DenseVector[Double]): Unit = {

    val bestCandidate = 0.0
    val expectedImprovement = new ExpectedImprovement(bestCandidate)
    val predicted = expectedImprovement(mu, sigma)

    assertIterableEqualsWithTolerance(predicted.toArray, expectedResult.toArray, TOL)
  }
}
