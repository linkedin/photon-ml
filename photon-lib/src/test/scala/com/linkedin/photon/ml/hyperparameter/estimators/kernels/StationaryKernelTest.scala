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
package com.linkedin.photon.ml.hyperparameter.estimators.kernels

import breeze.linalg.{DenseMatrix, DenseVector}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance

/**
 * Test cases for the StationaryKernel class
 */
class StationaryKernelTest {

  private val tol = 1e-7
  private val kernel = new RBF(
    noise = 0.0,
    indexedTransformMap = Map(1 -> ((x:Double) => Math.pow(10, x)), 3 -> ((x:Double) => Math.pow(x, 2))))

  /**
   * Test data and results generated from reference implementation in scikit-learn
   */
  @DataProvider
  def logLikelihoodProvider() =
    Array(
      Array(
        DenseMatrix((1.0, 2.0), (3.0, 4.0)),
        DenseVector(1.0, 0.0),
        DenseVector(1.0, 0.0, 1.0),
        -2.3378770946057106),
      Array(
        DenseMatrix((0.01388442, -0.45110147, -1.19551816,  0.67570627),
          (0.03187212,  0.26128739,  0.09664432,  0.51288568),
          (-0.1892356 , -0.37331575, -0.63040424, -0.91800163),
          (-0.41915027, -0.63310943, -0.33619354, -1.54669407)),
        DenseVector(1.97492722, -0.14301486, -0.17681195, -0.25272319),
        DenseVector(1.0, 0.0, 1.0),
        -5.6170735015658586),
      Array(
        DenseMatrix((-0.80210875,  1.01741171,  0.14846267,  1.15931876),
          (1.40446672, -0.06407389, -0.06608613, -0.55499189),
          (0.65113077, -0.53180815, -0.51595562,  0.08615354),
          (0.33328126,  0.89654475,  0.99865134, -0.34262719)),
        DenseVector(0.01540465,  0.93196629, -0.83929026,  0.39678908),
        DenseVector(1.0, 0.0, 0.36787944117144233),
        -4.5455256227943401),
      Array(
        DenseMatrix((0.32817291, -0.62739075, -0.15141223),
          (-0.33697839, -0.49970007, -0.30290632),
          (-0.49786383, 0.34232845, 0.11775675),
          (-0.86069848, -0.60832783, 0.13357631)),
        DenseVector(0.16192959, 0.86015525, -0.14415703, -0.46888909),
        DenseVector(1.0, 0.0, 1.0),
        -6.0655926036008498))

  @Test(dataProvider = "logLikelihoodProvider")
  def testLogLikelihood(
      x: DenseMatrix[Double],
      y: DenseVector[Double],
      theta: DenseVector[Double],
      expectedLikelihood: Double): Unit = {

    val likelihood = kernel.withParams(theta).logLikelihood(x, y)
    assertEquals(likelihood, expectedLikelihood, tol)
  }

  /**
    * Test for that the unwrap function can transform elements correctly given indexedTransformMap in the kernel
    */
  @Test
  def testUnwrap(): Unit = {

    val x = DenseMatrix((1.0, 3.0, 2.0, 4.0),
      (0.0, 2.0, 3.6, -1.0),
      (10.0, -3.0, 0.0, -3.0))
    val u = DenseMatrix((1.0, 1000.0, 2.0, 16.0),
      (0.0, 100.0, 3.6, 1.0),
      (10.0, 0.001, 0.0, 9.0))

    assertEquals(kernel.transform(x), u)
  }
}
