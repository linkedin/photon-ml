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
package com.linkedin.photon.ml.hyperparameter

import breeze.linalg.DenseVector
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.HyperparameterTuningMode
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Unit tests for [[VectorRescaling]].
 */
class VectorRescalingTest {

  /**
   * Unit test for VectorRescaling.transformForward
   */
  @Test
  def testTransformForward(): Unit = {
    val vector = DenseVector(1000, 0.001, 8, 4)
    val transformMap = Map(0 -> "LOG", 1 -> "LOG", 3-> "SQRT")
    val vectorTransformed = VectorRescaling.transformForward(vector, transformMap)
    val expectedData = DenseVector(3.0, -3.0, 8.0, 2.0)
    assertEquals(vectorTransformed, expectedData)
  }

  /**
   * Unit test for VectorRescaling.transformBackward
   */
  @Test
  def testTransformBackward(): Unit = {
    val vector = DenseVector(3.0, -3.0, 8.0, 2.0)
    val transformMap = Map(0 -> "LOG", 1 -> "LOG", 3-> "SQRT")
    val vectorTransformed = VectorRescaling.transformBackward(vector, transformMap)
    val expectedData = DenseVector(1000, 0.001, 8, 4)
    assertEquals(vectorTransformed, expectedData)
  }

  /**
   * Unit test for VectorRescaling.scaleForward
   */
  @Test
  def testScaleForward(): Unit = {

    val vector = DenseVector(5, 0.5, -1.0, 10.23)
    val ranges = Seq(DoubleRange(4, 11), DoubleRange(0.01, 0.99), DoubleRange(-2, 2), DoubleRange(-3, 3))
    val discreteParam = Map(0 -> 8)
    val vectorScaled = VectorRescaling.scaleForward(vector, ranges, discreteParam.keySet)
    val expectedData = DenseVector(0.125, 0.5, 0.25, 2.205)
    assertEquals(vectorScaled, expectedData)
  }

  /**
   * Unit test for VectorRescaling.scaleBackward
   */
  @Test
  def testScaleBackward(): Unit = {

    val vector = DenseVector(0.125, 0.5, 0.25, 2.205)
    val ranges = Seq(DoubleRange(4, 11), DoubleRange(0.01, 0.99), DoubleRange(-2, 2), DoubleRange(-3, 3))
    val discreteParam = Map(0 -> 8)
    val vectorScaled = VectorRescaling.scaleBackward(vector, ranges, discreteParam.keySet)
    val expectedData = DenseVector(5, 0.5, -1.0, 10.23)
    assertEquals(vectorScaled, expectedData)
  }

  /**
   * Unit test for VectorRescaling.rescalePriors
   */
  @Test
  def testRescalePriors(): Unit = {

    val tuningMode = HyperparameterTuningMode.BAYESIAN
    val hyperparameters = Seq("alpha", "beta", "gamma", "lambda")
    val ranges = Seq(DoubleRange(0, 4), DoubleRange(0, 4), DoubleRange(-2, 2), DoubleRange(-2, 2))
    val discreteParam = Map(0 -> 8)
    val transformMap = Map(0 -> "LOG", 1 -> "LOG", 3-> "SQRT")

    val hyperParams = HyperparameterConfig(tuningMode, hyperparameters, ranges, discreteParam, transformMap)

    val priors = Seq((DenseVector(1000.0, 1000.0, 8.0, 4.0), 0.1))

    val priorsRescaled = VectorRescaling.rescalePriors(priors, hyperParams)
    val expectedData = Seq((DenseVector(0.6, 0.75, 2.5, 1), 0.1))

    assertEquals(priorsRescaled, expectedData)
  }
}
