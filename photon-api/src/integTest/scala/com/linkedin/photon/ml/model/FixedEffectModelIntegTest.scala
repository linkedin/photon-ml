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
package com.linkedin.photon.ml.model

import scala.util.Random

import breeze.linalg.DenseVector
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * Test the fixed effect model.
 */
class FixedEffectModelIntegTest extends SparkTestUtils {

  @Test
  def testEquals(): Unit = sparkTest("testEqualsForFixedEffectModel") {

    // Coefficients parameter
    val coefficientDimension = 1
    val glm: GeneralizedLinearModel =
      LogisticRegressionModel(Coefficients.initializeZeroCoefficients(coefficientDimension))

    // Meta data
    val featureShardId = "featureShardId"

    // Should equal to itself
    val fixedEffectModel = new FixedEffectModel(sc.broadcast(glm), featureShardId)
    assertEquals(fixedEffectModel, fixedEffectModel)

    // Should equal to the fixed effect model with same featureShardId and model
    val fixedEffectModelCopy = new FixedEffectModel(sc.broadcast(glm), featureShardId)
    assertEquals(fixedEffectModel, fixedEffectModelCopy)

    // Should not equal to the fixed effect model with different featureShardId
    val differentFeatureShardId = "differentFeatureShardId"
    val fixedEffectModelWithDiffFeatureShardId = new FixedEffectModel(sc.broadcast(glm), differentFeatureShardId)
    assertNotEquals(fixedEffectModel, fixedEffectModelWithDiffFeatureShardId)

    // Should not equal to the fixed effect model with different model
    val differentCoefficientDimension = coefficientDimension + 1
    val differentGLM: GeneralizedLinearModel =
      LogisticRegressionModel(Coefficients.initializeZeroCoefficients(differentCoefficientDimension))

    val fixedEffectModelWithDiffCoefficientsRDD = new FixedEffectModel(sc.broadcast(differentGLM), featureShardId)
    assertNotEquals(fixedEffectModel, fixedEffectModelWithDiffCoefficientsRDD)
  }

  @Test
  def testEquals2(): Unit = sparkTest("testEqualsForFixedEffectModel") {

    // Coefficients parameter
    val numCoefficients = 10
    val randMeans = DenseVector.apply[Double](Seq.fill(numCoefficients)(Random.nextDouble).toArray)
    val coefficients = new Coefficients(randMeans, None) // no variance, which is not loaded right now
    val glm: GeneralizedLinearModel = LogisticRegressionModel(coefficients)

    // Meta data
    val featureShardId = "featureShardId"

    // Should equal to itself
    val fixedEffectModel = new FixedEffectModel(sc.broadcast(glm), featureShardId)
    assertEquals(fixedEffectModel, fixedEffectModel)

    // Should equal to the fixed effect model with same featureShardId and model
    val fixedEffectModelCopy = new FixedEffectModel(sc.broadcast(glm), featureShardId)
    assertEquals(fixedEffectModel, fixedEffectModelCopy)

    // Should not equal to the fixed effect model with different featureShardId
    val differentFeatureShardId = "differentFeatureShardId"
    val fixedEffectModelWithDiffFeatureShardId = new FixedEffectModel(sc.broadcast(glm), differentFeatureShardId)
    assertNotEquals(fixedEffectModel, fixedEffectModelWithDiffFeatureShardId)

    // Should not equal to the fixed effect model with different model
    val otherCoeffs = Seq.fill(numCoefficients)(Random.nextDouble)
    val randMeans2 = DenseVector.apply[Double](otherCoeffs.toArray)
    val coefficients2 = new Coefficients(randMeans2, None) // no variance, which is not loaded right now
    val differentGLM: GeneralizedLinearModel = LogisticRegressionModel(coefficients2)

    val fixedEffectModelWithDiffCoefficientsRDD = new FixedEffectModel(sc.broadcast(differentGLM), featureShardId)
    assertNotEquals(fixedEffectModel, fixedEffectModelWithDiffCoefficientsRDD)

    // Should not equal to the fixed effect model with different model
    val randMeans3 = DenseVector.apply[Double](otherCoeffs.take(5).toArray)
    val coefficients3 = new Coefficients(randMeans3, None) // no variance, which is not loaded right now
    val differentGLM3: GeneralizedLinearModel = LogisticRegressionModel(coefficients3)

    val fixedEffectModelWithDiffCoefficientsRDD3 = new FixedEffectModel(sc.broadcast(differentGLM3), featureShardId)
    assertNotEquals(fixedEffectModel, fixedEffectModelWithDiffCoefficientsRDD3)
  }
}
