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

import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.optimization.LogisticRegressionOptimizationProblem
import com.linkedin.photon.ml.test.SparkTestUtils


/**
 * Test the random effect model
 */
class RandomEffectModelTest extends SparkTestUtils {

  @Test
  def testEquals(): Unit = sparkTest("testEqualsForRandomEffectModel") {

    // Coefficients parameter
    val coefficientDimension = 1
    val glm: GeneralizedLinearModel = LogisticRegressionOptimizationProblem.initializeZeroModel(coefficientDimension)

    // Meta data
    val featureShardId = "featureShardId"
    val randomEffectId = "randomEffectId"

    // Random effect model
    val numCoefficients = 5
    val modelsRDD = sc.parallelize(Seq.tabulate(numCoefficients)(i => (i.toString, glm)))

    val randomEffectModel = new RandomEffectModel(modelsRDD, randomEffectId, featureShardId)

    // Should equal to itself
    assertEquals(randomEffectModel, randomEffectModel)

    // Should equal to the random effect model with same featureShardId, randomEffectId and coefficientsRDD
    val randomEffectModelCopy = new RandomEffectModel(modelsRDD, randomEffectId, featureShardId)
    assertEquals(randomEffectModel, randomEffectModelCopy)

    // Should not equal to the random effect model with different featureShardId
    val featureShardId1 = "featureShardId1"
    val randomEffectModelWithDiffFeatureShardId =
      new RandomEffectModel(modelsRDD, randomEffectId, featureShardId1)
    assertNotEquals(randomEffectModel, randomEffectModelWithDiffFeatureShardId)

    // Should not equal to the random effect model with different randomEffectId
    val randomEffectId1 = "randomEffectId1"
    val randomEffectModelWithDiffRandomEffectShardId =
      new RandomEffectModel(modelsRDD, randomEffectId1, featureShardId)
    assertNotEquals(randomEffectModel, randomEffectModelWithDiffRandomEffectShardId)

    // Should not equal to the random effect model with different coefficientsRDD
    val numCoefficients1 = numCoefficients + 1
    val modelsRDD1 = sc.parallelize(Seq.tabulate(numCoefficients1)(i => (i.toString, glm)))

    val randomEffectModelWithDiffCoefficientsRDD =
      new RandomEffectModel(modelsRDD1, randomEffectId, featureShardId)
    assertNotEquals(randomEffectModel, randomEffectModelWithDiffCoefficientsRDD)
  }
}
