/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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

import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * Test the fixed effect model
 */
class FixedEffectModelTest extends SparkTestUtils {

  @Test
  def testEquals(): Unit = sparkTest("testEqualsForFixedEffectModel") {
    // Coefficients parameter
    val coefficientDimension = 1
    val glm: GeneralizedLinearModel =
      LogisticRegressionModel.create(Coefficients.initializeZeroCoefficients(coefficientDimension))

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
      LogisticRegressionModel.create(Coefficients.initializeZeroCoefficients(differentCoefficientDimension))

    val fixedEffectModelWithDiffCoefficientsRDD = new FixedEffectModel(sc.broadcast(differentGLM), featureShardId)
    assertNotEquals(fixedEffectModel, fixedEffectModelWithDiffCoefficientsRDD)
  }
}
