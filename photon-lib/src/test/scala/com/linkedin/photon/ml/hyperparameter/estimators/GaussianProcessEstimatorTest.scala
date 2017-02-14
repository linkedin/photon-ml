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
package com.linkedin.photon.ml.hyperparameter.estimators

import scala.math.sin

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.linalg.linspace
import breeze.numerics.pow
import breeze.stats.mean
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.hyperparameter.estimators.kernels.RBF

/**
 * Test cases for the GaussianProcessEstimator class
 */
class GaussianProcessEstimatorTest {

  private val seed = 0
  private val estimator = new GaussianProcessEstimator(kernel = new RBF, seed = seed)
  private val tol = 1e-7

  /**
   * Test data and results generated from reference implementation in scikit-learn
   */
  @DataProvider
  def logLikelihoodProvider() =
    Array(
      Array(
        DenseMatrix((1.0, 2.0), (3.0, 4.0)),
        DenseVector(1.0, 0.0),
        DenseVector(0.0),
        -2.3378770946057106),
      Array(
        DenseMatrix((0.01388442, -0.45110147, -1.19551816,  0.67570627),
          (0.03187212,  0.26128739,  0.09664432,  0.51288568),
          (-0.1892356 , -0.37331575, -0.63040424, -0.91800163),
          (-0.41915027, -0.63310943, -0.33619354, -1.54669407)),
        DenseVector(1.97492722, -0.14301486, -0.17681195, -0.25272319),
        DenseVector(0.0),
        -5.6170735015658586),
      Array(
        DenseMatrix((-0.80210875,  1.01741171,  0.14846267,  1.15931876),
          (1.40446672, -0.06407389, -0.06608613, -0.55499189),
          (0.65113077, -0.53180815, -0.51595562,  0.08615354),
          (0.33328126,  0.89654475,  0.99865134, -0.34262719)),
        DenseVector(0.01540465,  0.93196629, -0.83929026,  0.39678908),
        DenseVector(-1.0),
        -4.5455256227943401),
      Array(
        DenseMatrix((0.32817291, -0.62739075, -0.15141223),
          (-0.33697839, -0.49970007, -0.30290632),
          (-0.49786383, 0.34232845, 0.11775675),
          (-0.86069848, -0.60832783, 0.13357631)),
        DenseVector(0.16192959, 0.86015525, -0.14415703, -0.46888909),
        DenseVector(0.0),
        -6.0655926036008498))

  @Test(dataProvider = "logLikelihoodProvider")
  def testLogLikelihood(
      x: DenseMatrix[Double],
      y: DenseVector[Double],
      theta: DenseVector[Double],
      expectedLikelihood: Double): Unit = {

    val likelihood = estimator.logLikelihood(x, y, theta)
    assertEquals(likelihood, expectedLikelihood, tol)
  }

  @Test
  def testFit(): Unit = {
    val x = linspace(0, 10, 100)
    val y = x.map(i => i * sin(i))

    val obsPoints = List(5, 50, 35, 75, 78)
    val xTrain = DenseMatrix(obsPoints.map(i => x(i)):_*)
    val yTrain = DenseVector(obsPoints.map(i => y(i)):_*)

    val model = estimator.fit(xTrain, yTrain)

    val (means, vars) = model.predict(x.toDenseMatrix.t)

    val mse = mean(pow(y - means, 2))
    assertTrue(mse < 1.66)
  }
}
