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
 * Test cases for the Matern52 class
 */
class Matern52Test {

  private val tol = 1e-7
  private val kernel = new Matern52

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
        DenseMatrix((1.0, 0.03239932, 0.04173912),
          (0.03239932, 1.0, 0.07761498),
          (0.04173912, 0.07761498, 1.0))),
      Array(
        DenseMatrix((0.32817291, -0.62739075, -0.15141223),
          (-0.33697839, -0.49970007, -0.30290632),
          (-0.49786383, 0.34232845, 0.11775675),
          (-0.86069848, -0.60832783, 0.13357631)),
        DenseMatrix((1.0, 0.71067495, 0.36649838, 0.40439812),
          (0.71067495, 1.0, 0.55029418, 0.71297005),
          (0.36649838, 0.55029418, 1.0, 0.51385965),
          (0.40439812, 0.71297005, 0.51385965, 1.0))),
      Array(
        DenseMatrix((-0.40944433, 0.39704702, -0.48894766),
          (1.03282411, -1.0380654, 0.65404646),
          (1.21080337, 0.5587334, 0.59055366),
          (1.33081, 1.20478412, 0.8560233)),
        DenseMatrix((1.0, 0.08284709, 0.14862395,0.08162984),
          (0.08284709, 1.0, 0.24441232, 0.09136301),
          (0.14862395, 0.24441232, 1.0, 0.70149793),
          (0.08162984, 0.09136301, 0.70149793, 1.0))))

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
        DenseMatrix((0.36431909, 0.44333958, 0.22917335, 0.08481237),
          (0.57182815, 0.19854279, 0.12340393, 0.04963231),
          (0.75944682, 0.11384187, 0.19003345, 0.10995123),
          (0.38353084, 0.13654483, 0.07208932, 0.03096713))),
      Array(
        DenseMatrix((0.32817291, -0.62739075, -0.15141223),
          (-0.33697839, -0.49970007, -0.30290632),
          (-0.49786383,  0.34232845,  0.11775675),
          (-0.86069848, -0.60832783,  0.13357631)),
        DenseMatrix((-0.92499106, 0.34302631, 0.84799782),
          (-0.83857738, -1.20129995,  0.06613189),
          (-1.6107072 , -0.8280462 ,  0.52490887),
          (-0.30898909, -1.13793004, -1.34480429)),
        DenseMatrix((0.16726349, 0.35900106, 0.12602633, 0.30430555),
          (0.26721572, 0.56024858, 0.26314172, 0.40466601),
          (0.61601685, 0.25344041, 0.2255722, 0.1210904),
          (0.42003005, 0.77070301, 0.59884283, 0.22590494))))

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
