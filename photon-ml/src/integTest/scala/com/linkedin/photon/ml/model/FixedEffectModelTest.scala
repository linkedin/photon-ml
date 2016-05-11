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

import org.testng.annotations.Test
import org.testng.Assert._

import com.linkedin.photon.ml.test.SparkTestUtils


/**
 * Test the fixed effect model
 */
class FixedEffectModelTest extends SparkTestUtils {

  @Test
  def testEquals() = sparkTest("testEqualsForFixedEffectModel") {
    // Coefficients parameter
    val coefficientDimension = 1
    val coefficients = Coefficients.initializeZeroCoefficients(coefficientDimension)

    // Meta data
    val featureShardId = "featureShardId"

    // Fixed effect model
    val fixedEffectModel = new FixedEffectModel(sc.broadcast(coefficients), featureShardId)

    // Should equal to itself
    assertEquals(fixedEffectModel, fixedEffectModel)

    // Should equal to the fixed effect model with same featureShardId and coefficientsBroadcast
    val fixedEffectModelCopy = new FixedEffectModel(sc.broadcast(coefficients), featureShardId)
    assertEquals(fixedEffectModel, fixedEffectModelCopy)

    // Should not equal to the fixed effect model with different featureShardId
    val featureShardId1 = "featureShardId1"
    val fixedEffectModelWithDiffFeatureShardId = new FixedEffectModel(sc.broadcast(coefficients), featureShardId1)
    assertNotEquals(fixedEffectModel, fixedEffectModelWithDiffFeatureShardId)

    // Should not equal to the fixed effect model with different coefficientsBroadcast
    val coefficientDimension1 = coefficientDimension + 1
    val coefficients1 = Coefficients.initializeZeroCoefficients(coefficientDimension1)
    val fixedEffectModelWithDiffCoefficientsRDD = new FixedEffectModel(sc.broadcast(coefficients1), featureShardId)
    assertNotEquals(fixedEffectModel, fixedEffectModelWithDiffCoefficientsRDD)
  }
}
