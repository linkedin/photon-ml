///*
// * Copyright 2017 LinkedIn Corp. All rights reserved.
// * Licensed under the Apache License, Version 2.0 (the "License"); you may
// * not use this file except in compliance with the License. You may obtain a
// * copy of the License at
// *
// * http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations
// * under the License.
// */
//package com.linkedin.photon.ml.model
//
//import org.testng.Assert._
//import org.testng.annotations.Test
//
//import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
//import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
//import com.linkedin.photon.ml.supervised.regression.PoissonRegressionModel
//import com.linkedin.photon.ml.test.SparkTestUtils
//
///**
// * Integration tests for [[RandomEffectModel]].
// */
//class RandomEffectModelIntegTest extends SparkTestUtils {
//
//  /**
//   * Test that a [[RandomEffectModel]] must have the same coefficients, be computed on the same feature shard, and have
//   * the same random effect type to be equal.
//   */
//  @Test
//  def testEquals(): Unit = sparkTest("testEqualsForRandomEffectModel") {
//    // Coefficients parameter
//    val coefficientDimension = 1
//    val glm: GeneralizedLinearModel =
//      LogisticRegressionModel(Coefficients.initializeZeroCoefficients(coefficientDimension))
//
//    // Meta data
//    val featureShardId = "featureShardId"
//    val randomEffectType = "randomEffectType"
//
//    // Random effect model
//    val numCoefficients = 5
//    val modelsRDD = sc.parallelize(Seq.tabulate(numCoefficients)(i => (i.toString, glm)))
//
//    val randomEffectModel = new RandomEffectModel(modelsRDD, randomEffectType, featureShardId)
//
//    // Should equal to itself
//    assertEquals(randomEffectModel, randomEffectModel)
//
//    // Should equal to the random effect model with same featureShardId, randomEffectType and coefficientsRDD
//    val randomEffectModelCopy = new RandomEffectModel(modelsRDD, randomEffectType, featureShardId)
//    assertEquals(randomEffectModel, randomEffectModelCopy)
//
//    // Should not equal to the random effect model with different featureShardId
//    val featureShardId1 = "featureShardId1"
//    val randomEffectModelWithDiffFeatureShardId =
//      new RandomEffectModel(modelsRDD, randomEffectType, featureShardId1)
//    assertNotEquals(randomEffectModel, randomEffectModelWithDiffFeatureShardId)
//
//    // Should not equal to the random effect model with different randomEffectType
//    val randomEffectType1 = "randomEffectType1"
//    val randomEffectModelWithDiffRandomEffectShardId =
//      new RandomEffectModel(modelsRDD, randomEffectType1, featureShardId)
//    assertNotEquals(randomEffectModel, randomEffectModelWithDiffRandomEffectShardId)
//
//    // Should not equal to the random effect model with different coefficientsRDD
//    val numCoefficients1 = numCoefficients + 1
//    val modelsRDD1 = sc.parallelize(Seq.tabulate(numCoefficients1)(i => (i.toString, glm)))
//
//    val randomEffectModelWithDiffCoefficientsRDD =
//      new RandomEffectModel(modelsRDD1, randomEffectType, featureShardId)
//    assertNotEquals(randomEffectModel, randomEffectModelWithDiffCoefficientsRDD)
//  }
//
//  /**
//   * Test that a [[RandomEffectModel]] consisting of the same type of [[GeneralizedLinearModel]] will be accepted.
//   */
//  @Test
//  def testModelsConsistencyGood(): Unit = sparkTest("testModelsConsistencyGood") {
//
//    val numFeatures = 10
//
//    // Random effect with 2 items of the same type.
//    val randomEffectItem1 = CoefficientsTest.sparseCoefficients(numFeatures)(1,5,7)(111,511,911)
//    val glm1: GeneralizedLinearModel = new LogisticRegressionModel(randomEffectItem1)
//    val randomEffectItem2 = CoefficientsTest.sparseCoefficients(numFeatures)(1,2)(112,512)
//    val glm2: GeneralizedLinearModel = new LogisticRegressionModel(randomEffectItem2)
//    val randomEffectRDD = sc.parallelize(List(("RandomEffectItem1", glm1), ("RandomEffectItem2", glm2)))
//
//    // This should not throw exception.
//    new RandomEffectModel(randomEffectRDD, "RandomEffectModel", "RandomEffectFeatures")
//  }
//
//  /**
//   * Test that a [[RandomEffectModel]] consisting of different types of [[GeneralizedLinearModel]] will be rejected.
//   */
//  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
//  def testModelsConsistencyBad(): Unit = sparkTest("testModelsConsistencyBad") {
//
//    val numFeatures = 10
//
//    // Random effect with 2 items of differing types.
//    val randomEffectItem1 = CoefficientsTest.sparseCoefficients(numFeatures)(1,5,7)(111,511,911)
//    val glm1: GeneralizedLinearModel = new LogisticRegressionModel(randomEffectItem1)
//    val randomEffectItem2 = CoefficientsTest.sparseCoefficients(numFeatures)(1,2)(112,512)
//    val glm2: GeneralizedLinearModel = new PoissonRegressionModel(randomEffectItem2)
//    val randomEffectRDD = sc.parallelize(List(("RandomEffectItem1", glm1), ("RandomEffectItem2", glm2)))
//
//    // This should throw exception.
//    new RandomEffectModel(randomEffectRDD, "RandomEffectModel", "RandomEffectFeatures")
//  }
//}
