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

import com.linkedin.photon.ml.hyperparameter.estimators.kernels.Matern52

/**
 * Test cases for the GaussianProcessEstimator class
 */
class GaussianProcessEstimatorTest {

  private val seed = 0L
  private val estimator = new GaussianProcessEstimator(
    kernel = new Matern52(noise = 0.0),
    seed = seed)
  private val tol = 1e-7

  @Test
  def testFit(): Unit = {
    val x = linspace(0, 10, 100)
    val y = x.map(i => i * sin(i))

    val obsPoints = List(5, 15, 35, 50, 75, 78, 90)
    val xTrain = DenseMatrix(obsPoints.map(i => x(i)):_*)
    val yTrain = DenseVector(obsPoints.map(i => y(i)):_*)

    val model = estimator.fit(xTrain, yTrain)

    val (means, vars) = model.predict(x.toDenseMatrix.t)

    val mse = mean(pow(y - means, 2))
    assertTrue(mse < 2.0)
  }
}
