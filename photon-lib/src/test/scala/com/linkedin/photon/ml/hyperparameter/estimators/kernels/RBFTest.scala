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
package com.linkedin.photon.ml.hyperparameter.estimators.kernels

import breeze.linalg.{DenseMatrix, DenseVector}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance

/**
 * Test cases for the RBF class
 */
class RBFTest {

  private val tol = 1e-7
  private val kernel = new RBF

  /**
   * Test data and results generated from reference implementation in scikit-learn
   */
  @DataProvider
  def kernelSourceProvider() =
    Array(
      Array(
        DenseMatrix((1.16629448, 2.06716533, -0.92010277),
          (0.32491615, -0.50086458, 0.15349931),
          (-1.29952204, 1.22238724, -0.0238411)),
        DenseMatrix((1.0, 0.01458651, 0.02240227),
          (0.01458651, 1.0, 0.05961054),
          (0.02240227, 0.05961054, 1.0))),
      Array(
        DenseMatrix((0.32817291, -0.62739075, -0.15141223),
          (-0.33697839, -0.49970007, -0.30290632),
          (-0.49786383, 0.34232845, 0.11775675),
          (-0.86069848, -0.60832783, 0.13357631)),
        DenseMatrix((1.0, 0.78596674, 0.42845397, 0.47354965),
          (0.78596674, 1.0, 0.63386024, 0.78796634),
          (0.42845397, 0.63386024, 1.0, 0.59581605),
          (0.47354965, 0.78796634, 0.59581605, 1.0))),
      Array(
        DenseMatrix((-0.40944433, 0.39704702, -0.48894766),
          (1.03282411, -1.0380654, 0.65404646),
          (1.21080337, 0.5587334, 0.59055366),
          (1.33081, 1.20478412, 0.8560233)),
        DenseMatrix((1.0, 0.06567344, 0.14832728, 0.06425244),
          (0.06567344, 1.0, 0.27451835, 0.07577536),
          (0.14832728, 0.27451835, 1.0, 0.7779223),
          (0.06425244, 0.07577536, 0.7779223,  1.0))))

  /**
   * Test data and results generated from reference implementation in scikit-learn
   */
  @DataProvider
  def kernelTwoSourceProvider() =
    Array(
      Array(
        DenseMatrix((0.32817291, -0.62739075, -0.15141223),
          (-0.33697839, -0.49970007, -0.30290632),
          (-0.49786383, 0.34232845, 0.11775675),
          (-0.86069848, -0.60832783, 0.13357631)),
        DenseMatrix((-0.40944433, 0.39704702, -0.48894766),
          (1.03282411, -1.0380654, 0.65404646),
          (1.21080337, 0.5587334, 0.59055366),
          (1.33081, 1.20478412, 0.8560233)),
        DenseMatrix((0.42581894, 0.518417, 0.25455962, 0.06798038),
          (0.65572813, 0.21417167, 0.11566118, 0.02974926),
          (0.82741311, 0.10351387, 0.2029175, 0.09862232),
          (0.44889221, 0.13258973, 0.0533442, 0.0134873))),
      Array(
        DenseMatrix((0.32817291, -0.62739075, -0.15141223),
          (-0.33697839, -0.49970007, -0.30290632),
          (-0.49786383,  0.34232845,  0.11775675),
          (-0.86069848, -0.60832783,  0.13357631)),
        DenseMatrix((-0.92499106, 0.34302631, 0.84799782),
          (-0.83857738, -1.20129995,  0.06613189),
          (-1.6107072 , -0.8280462 ,  0.52490887),
          (-0.30898909, -1.13793004, -1.34480429)),
        DenseMatrix((0.17282516, 0.41936999, 0.11901991, 0.35154935),
          (0.30414111,  0.64402575,  0.29887283,  0.47386342),
          (0.69918137,  0.28628435,  0.24982739,  0.11270722),
          (0.49174098,  0.83666878,  0.6825188 ,  0.25026486))))

  @Test(dataProvider = "kernelSourceProvider")
  def testKernelApply(
      x: DenseMatrix[Double],
      expectedResult: DenseMatrix[Double]): Unit = {

    val result = kernel(x)
    assertIterableEqualsWithTolerance(result.toArray, expectedResult.toArray, tol)
  }

  @Test(dataProvider = "kernelTwoSourceProvider")
  def testKernelTwoSourceApply(
      x1: DenseMatrix[Double],
      x2: DenseMatrix[Double],
      expectedResult: DenseMatrix[Double]): Unit = {

    val result = kernel(x1, x2)
    assertIterableEqualsWithTolerance(result.toArray, expectedResult.toArray, tol)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testEmptyInput(): Unit = {
    kernel(DenseMatrix.zeros[Double](0, 0))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testDimensionMismatch(): Unit = {
    kernel(DenseMatrix.zeros[Double](2, 2), DenseMatrix.zeros[Double](2, 3))
  }
}
