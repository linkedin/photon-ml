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
package com.linkedin.photon.ml.optimization

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

class OptimizationUtilsTest {
  @DataProvider
  def generateCoefficientsAndConstraintMap(): Array[Array[Object]] = {
    val dVec = DenseVector(0.0, 1.0, -1.0, 0.0, 5.0)
    val sVec = SparseVector(0.0, 1.0, -1.0, 0.0, 5.0)

    Array(
      Array(dVec, None, dVec),
      Array(sVec, None, sVec),
      Array(dVec, Some(Map[Int, (Double, Double)]()), dVec),
      Array(sVec, Some(Map[Int, (Double, Double)]()), sVec),
      Array(dVec, Some(Map[Int, (Double, Double)](1->(-0.5, 0.5), 4->(6.7, Double.PositiveInfinity))),
        DenseVector(0.0, 0.5, -1.0, 0.0, 6.7)),
      Array(sVec, Some(Map[Int, (Double, Double)](1->(-0.5, 0.5), 4->(6.7, Double.PositiveInfinity))),
        SparseVector(0.0, 0.5, -1.0, 0.0, 6.7)),
      Array(dVec,
        Some(Map[Int, (Double, Double)](0->(-1.0, 0.0), 1->(-0.5, 0.5), 2->(0.0, 1.0), 3->(Double.NegativeInfinity, 0.0),
          4->(6.7, Double.PositiveInfinity))),
        DenseVector(0.0, 0.5, 0.0, 0.0, 6.7)),
      Array(sVec,
        Some(Map[Int, (Double, Double)](0->(-1.0, 0.0), 1->(-0.5, 0.5), 2->(0.0, 1.0), 3->(Double.NegativeInfinity, 0.0),
          4->(6.7, Double.PositiveInfinity))),
        SparseVector(0.0, 0.5, 0.0, 0.0, 6.7))
    )
  }

  @Test(dataProvider = "generateCoefficientsAndConstraintMap")
  def testProjectCoefficientsToHypercube(
      coefficients: Vector[Double],
      constraints: Option[Map[Int, (Double, Double)]],
      expectedVectorOutput: Vector[Double]): Unit =
    Assert.assertEquals(
      OptimizationUtils.projectCoefficientsToHypercube(coefficients, constraints),
      expectedVectorOutput)
}
