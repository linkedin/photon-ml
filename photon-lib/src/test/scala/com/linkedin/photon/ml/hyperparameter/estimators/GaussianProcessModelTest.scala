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

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sqrt
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.hyperparameter.estimators.kernels.RBF
import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance

/**
 * Test cases for the GaussianProcessModel class
 */
class GaussianProcessModelTest {
  private val tol = 1e-7
  private val kernel = new RBF(noise = 0.0, lengthScale = DenseVector(1.0))

  /**
   * Test data and results generated from reference implementation in scikit-learn
   */
  @DataProvider
  def predictionProvider() =
    Array(
      Array(
        DenseMatrix((0.00773725, -0.31298875, 0.27183008),
          (-0.68440447, -0.8561772, -0.78500855),
          (-0.02330709, -1.92979733, 0.43287544),
          (-0.85140297, -1.49877559, -1.63778668)),
        DenseVector(-0.34459489, -0.0485107, -1.29375589, 1.11622403),
        DenseMatrix((-0.31800735, 1.34422005, -1.55408361),
          (-0.60237846, -1.00816597, -0.09440482),
          ( 0.31517342, -1.11984756, -0.9466699 ),
          ( 0.11024813, -1.43619905, 0.67390101)),
        DenseVector(-0.01325603, -0.66403465, -0.10878228, -1.10488029),
        DenseVector(0.99747502, 0.44726687, 0.79425794, 0.44201904)),
      Array(
        DenseMatrix((0.69567278, -0.41581942, 0.85500744),
          ( 0.98204282, -0.29115782, -0.22831259),
          (-0.46622083, -0.68199927, -0.09467517),
          ( 0.12449017, -0.37616456, -0.27992044)),
        DenseVector(-0.11453575,  0.95807664, -0.7181996 , -0.29513717),
        DenseMatrix(( 1.21362357,  0.18562891, -1.62395987),
          (-0.75193848, 0.48940236, -0.98794203),
          (-0.43582962, 1.83947234, 0.0808053 ),
          (-0.73004528, -1.83643245, -0.33303083)),
        DenseVector(0.46723757, -0.34857392, -0.05126064, -0.24301167),
        DenseVector(0.92967279, 0.91067249, 0.99688996, 0.83459746)),
      Array(
        DenseMatrix((-0.46055067, 0.93364116, -1.09573962),
          (-1.20787535, 0.33594068, -1.95753059),
          (-0.84306614, -0.6812687 , -0.74283257),
          (-0.95882761, 0.51132399, -0.13720216)),
        DenseVector(-0.98494485, 0.186753, -0.65985498, 0.52334382),
        DenseMatrix((-1.00757146, 0.78187748, -0.78197457),
          (1.52226612, 0.43348454, -1.31427541),
          (0.21296738, -0.77575617, 1.46077293),
          (0.35616412, -0.01987576, -1.05690365)),
        DenseVector(-0.16836956, -0.22862767, 0.04165401, -0.77207482),
        DenseVector(0.3791334, 0.99059374, 0.99728549, 0.83955005)))

  @Test(dataProvider = "predictionProvider")
  def testPredict(
      xTrain: DenseMatrix[Double],
      yTrain: DenseVector[Double],
      x: DenseMatrix[Double],
      expectedMeans: DenseVector[Double],
      expectedStd: DenseVector[Double]): Unit = {

    val model = new GaussianProcessModel(
      xTrain,
      yTrain,
      yMean = 0.0,
      kernels = Seq(kernel),
      predictionTransformation = None)

    val (means, vars) = model.predict(x)

    assertIterableEqualsWithTolerance(means.toArray, expectedMeans.toArray, tol)
    assertIterableEqualsWithTolerance(sqrt(vars).toArray, expectedStd.toArray, tol)
  }

  @Test(dataProvider = "predictionProvider")
  def testPredictTransformed(
      xTrain: DenseMatrix[Double],
      yTrain: DenseVector[Double],
      x: DenseMatrix[Double],
      expectedMeans: DenseVector[Double],
      expectedStd: DenseVector[Double]): Unit = {

    def trans(means: DenseVector[Double], std: DenseVector[Double]) =
      1.7 * means / std

    val transformation = new PredictionTransformation {
      def isMaxOpt: Boolean = true
      def apply(
          predictiveMeans: DenseVector[Double],
          predictiveVariances: DenseVector[Double]): DenseVector[Double] =
        trans(predictiveMeans, sqrt(predictiveVariances))
    }

    val model = new GaussianProcessModel(
      xTrain,
      yTrain,
      yMean = 0.0,
      kernels = Seq(kernel),
      predictionTransformation = Some(transformation))

    val vals = model.predictTransformed(x)

    assertIterableEqualsWithTolerance(vals.toArray, trans(expectedMeans, expectedStd).toArray, tol)
  }
}
