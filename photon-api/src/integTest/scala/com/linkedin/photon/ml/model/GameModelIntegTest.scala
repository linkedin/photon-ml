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
//import org.apache.spark.SparkContext
//import org.testng.Assert._
//import org.testng.annotations.Test
//
//import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
//import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
//import com.linkedin.photon.ml.supervised.regression.PoissonRegressionModel
//import com.linkedin.photon.ml.test.SparkTestUtils
//
///**
// * Integration tests for [[GameModel]].
// */
//class GameModelIntegTest extends SparkTestUtils {
//
//  /**
//   * Generate a toy fixed effect model.
//   *
//   * @param sc The Spark context
//   * @param coefficientDimension The dimension of the coefficients
//   * @return A fixed effect model
//   */
//  protected def getFixedEffectModel(sc: SparkContext, coefficientDimension: Int): FixedEffectModel = {
//
//    // Coefficients parameter
//    val glm: GeneralizedLinearModel =
//      LogisticRegressionModel(Coefficients.initializeZeroCoefficients(coefficientDimension))
//
//    // Meta data
//    val featureShardId = "featureShardId"
//
//    // Fixed effect model
//    new FixedEffectModel(sc.broadcast(glm), featureShardId)
//  }
//
//  /**
//   * Generate a toy random effect model.
//   *
//   * @param sc The Spark context
//   * @param coefficientDimension The dimension of the coefficients
//   * @return A random effect model
//   */
//  protected def getRandomEffectModel(sc: SparkContext, coefficientDimension: Int): RandomEffectModel = {
//
//    // Coefficients parameter
//    val glm: GeneralizedLinearModel =
//      LogisticRegressionModel(Coefficients.initializeZeroCoefficients(coefficientDimension))
//
//    // Meta data
//    val featureShardId = "featureShardId"
//    val REType = "REType"
//
//    // Random effect model
//    val numCoefficients = 5
//    val modelsRDD = sc.parallelize(Seq.tabulate(numCoefficients)(i => (i.toString, glm)))
//    new RandomEffectModel(modelsRDD, REType, featureShardId)
//  }
//
//  @Test
//  def testGetModel(): Unit = sparkTest("testGetModel") {
//
//    val FEModelName1 = "fix1"
//    val REModelName1 = "random1"
//    val FEModelName2 = "fix2"
//    val REModelName2 = "random2"
//
//    val FEModel1 = getFixedEffectModel(sc, 1)
//    val FEModel2 = getFixedEffectModel(sc, 2)
//    val REModel1 = getRandomEffectModel(sc, 1)
//    val REModel2 = getRandomEffectModel(sc, 2)
//
//    // case 1: fixed effect model only
//    val FEModelOnly = GameModel((FEModelName1, FEModel1), (FEModelName2, FEModel2))
//    assertEquals(FEModel1, FEModelOnly.getModel(FEModelName1).get)
//    assertEquals(FEModel2, FEModelOnly.getModel(FEModelName2).get)
//    assertTrue(FEModelOnly.getModel(REModelName1).isEmpty)
//
//    // case 2: random effect model only
//    val REModelOnly = GameModel((REModelName1, REModel1), (REModelName2, REModel2))
//    assertEquals(REModel1, REModelOnly.getModel(REModelName1).get)
//    assertEquals(REModel2, REModelOnly.getModel(REModelName2).get)
//    assertTrue(REModelOnly.getModel(FEModelName2).isEmpty)
//
//    // case 3: fixed and random effect model
//    val fixedAndRandomEffectModel = GameModel((FEModelName1, FEModel1), (REModelName2, REModel2))
//    assertEquals(FEModel1, fixedAndRandomEffectModel.getModel(FEModelName1).get)
//    assertEquals(REModel2, fixedAndRandomEffectModel.getModel(REModelName2).get)
//    assertTrue(fixedAndRandomEffectModel.getModel(FEModelName2).isEmpty)
//    assertTrue(fixedAndRandomEffectModel.getModel(REModelName1).isEmpty)
//  }
//
//  @Test
//  def testUpdateModelOfSameType(): Unit = sparkTest("testUpdateModelOfSameType") {
//
//    val FEModelName = "fix"
//    val REModelName = "random"
//
//    val FEModel1 = getFixedEffectModel(sc, 1)
//    val FEModel2 = getFixedEffectModel(sc, 2)
//    val REModel1 = getRandomEffectModel(sc, 1)
//    val REModel2 = getRandomEffectModel(sc, 2)
//
//    val gameModel11 = GameModel((FEModelName, FEModel1), (REModelName, REModel1))
//    assertEquals(gameModel11.getModel(FEModelName).get, FEModel1)
//    assertEquals(gameModel11.getModel(REModelName).get, REModel1)
//    val gameModel21 = gameModel11.updateModel(FEModelName, FEModel2)
//    assertEquals(gameModel21.getModel(FEModelName).get, FEModel2)
//    val gameModel22 = gameModel21.updateModel(REModelName, REModel2)
//    assertEquals(gameModel22.getModel(REModelName).get, REModel2)
//  }
//
//  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
//  def testUpdateModelOfDifferentType(): Unit = sparkTest("testUpdateModelOfDifferentType") {
//
//    val FEModelName = "fix"
//
//    val FEModel = getFixedEffectModel(sc, 1)
//    val REModel = getRandomEffectModel(sc, 1)
//
//    val gameModel = GameModel((FEModelName, FEModel))
//    gameModel.updateModel(FEModelName, REModel)
//  }
//
//  @Test
//  def testToMap(): Unit = sparkTest("testToMap") {
//
//    val FEModelName = "fix"
//    val REModelName = "random"
//
//    val FEModel = getFixedEffectModel(sc, 1)
//    val REModel = getRandomEffectModel(sc, 1)
//
//    val modelsMap = Map(FEModelName -> FEModel, REModelName -> REModel)
//    val gameModel = new GameModel(modelsMap)
//    assertEquals(gameModel.toMap, modelsMap)
//  }
//
//  @Test
//  def testEquals(): Unit = sparkTest("testEquals") {
//
//    val FEModelName1 = "fix1"
//    val REModelName1 = "random1"
//    val FEModelName2 = "fix2"
//    val REModelName2 = "random2"
//
//    val FEModel1 = getFixedEffectModel(sc, 1)
//    val FEModel2 = getFixedEffectModel(sc, 2)
//    val REModel1 = getRandomEffectModel(sc, 1)
//    val REModel2 = getRandomEffectModel(sc, 1)
//
//    val gameModel1111 = GameModel((FEModelName1, FEModel1), (REModelName1, REModel1))
//    val gameModel1112 = GameModel((FEModelName1, FEModel1), (REModelName1, REModel2))
//    val gameModel1212 = GameModel((FEModelName1, FEModel2), (REModelName1, REModel2))
//    val gameModel1122 = GameModel((FEModelName1, FEModel1), (REModelName2, REModel2))
//    val gameModel2121 = GameModel((FEModelName2, FEModel1), (REModelName2, REModel1))
//    val gameModel2211 = GameModel((FEModelName2, FEModel2), (REModelName1, REModel1))
//    val gameModel2212 = GameModel((FEModelName2, FEModel2), (REModelName1, REModel2))
//
//    // Same name and model
//    assertEquals(gameModel1111, gameModel1111)
//    assertEquals(gameModel1111, gameModel1112)
//    assertEquals(gameModel2211, gameModel2212)
//
//    // Either name or model is different
//    assertNotEquals(gameModel1212, gameModel1122)
//    assertNotEquals(gameModel2121, gameModel2211)
//    assertNotEquals(gameModel1212, gameModel2212)
//  }
//
//  @Test
//  def testModelsConsistencyGood(): Unit = sparkTest("testModelsConsistencyGood") {
//
//    // Features: we have three feature spaces: one for the fixed model, and one for each random model.
//    // Each model has its own separate feature space, but feature values can be shared between spaces.
//    // Features shared between spaces have a unique name, but possibly different indices.
//    val numFeaturesPerModel = Map(("fixedFeatures", 10), ("RE1Features", 10), ("RE2Features", 10))
//
//    // Fixed effect model
//    val glm = new LogisticRegressionModel(
//      CoefficientsTest.sparseCoefficients(numFeaturesPerModel("fixedFeatures"))(1,2,5)(11,21,51))
//    val FEModel = new FixedEffectModel(sc.broadcast(glm), "fixedFeatures")
//
//    // Random effect 1 has 2 items
//    val numFeaturesRE1 = numFeaturesPerModel("RE1Features")
//    val RE1Item1 = CoefficientsTest.sparseCoefficients(numFeaturesRE1)(1,5,7)(111,511,911)
//    val glmRE11: GeneralizedLinearModel = new LogisticRegressionModel(RE1Item1)
//    val RE1Item2 = CoefficientsTest.sparseCoefficients(numFeaturesRE1)(1,2)(112,512)
//    val glmRE12: GeneralizedLinearModel = new LogisticRegressionModel(RE1Item2)
//
//    val glmRE1RDD = sc.parallelize(List(("RE1Item1", glmRE11), ("RE1Item2", glmRE12)))
//    val RE1Model = new RandomEffectModel(glmRE1RDD, "REModel1", "RE1Features")
//
//    // Random effect 2 has 3 items (of a different kind)
//    val numFeaturesRE2 = numFeaturesPerModel("RE2Features")
//    val RE2Item1 = CoefficientsTest.sparseCoefficients(numFeaturesRE2)(3,4,6)(321,421,621)
//    val glmRE21: GeneralizedLinearModel = new LogisticRegressionModel(RE2Item1)
//    val RE2Item2 = CoefficientsTest.sparseCoefficients(numFeaturesRE2)(4,5)(322,422)
//    val glmRE22: GeneralizedLinearModel = new LogisticRegressionModel(RE2Item2)
//    val RE2Item3 = CoefficientsTest.sparseCoefficients(numFeaturesRE2)(2,7,8)(323,423,523)
//    val glmRE23: GeneralizedLinearModel = new LogisticRegressionModel(RE2Item3)
//
//    val glmRE2RDD = sc.parallelize(List(("RE2Item1", glmRE21), ("RE2Item2", glmRE22), ("RE2Item3", glmRE23)))
//    val RE2Model = new RandomEffectModel(glmRE2RDD, "REModel2", "RE2Features")
//
//    // This GAME model has 1 fixed effect, and 2 different random effect models
//    GameModel(("fixed", FEModel), ("RE1", RE1Model), ("RE2", RE2Model))
//  }
//
//  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
//  def testModelsConsistencyBad(): Unit = sparkTest("testModelsConsistencyBad") {
//
//    // Features: we have three feature spaces: one for the fixed model, and one for each random model.
//    // Each model has its own separate feature space, but feature values can be shared between spaces.
//    // Features shared between spaces have a unique name, but possibly different indices.
//    val numFeaturesPerModel = Map(("fixedFeatures", 10), ("RE1Features", 10), ("RE2Features", 10))
//
//    // Fixed effect model
//    val glm = new LogisticRegressionModel(
//      CoefficientsTest.sparseCoefficients(numFeaturesPerModel("fixedFeatures"))(1,2,5)(11,21,51))
//    val FEModel = new FixedEffectModel(sc.broadcast(glm), "fixedFeatures")
//
//    // Random effect 1 has 2 items
//    val numFeaturesRE1 = numFeaturesPerModel("RE1Features")
//    val RE1Item1 = CoefficientsTest.sparseCoefficients(numFeaturesRE1)(1,5,7)(111,511,911)
//    val glmRE11: GeneralizedLinearModel = new LogisticRegressionModel(RE1Item1)
//    val RE1Item2 = CoefficientsTest.sparseCoefficients(numFeaturesRE1)(1,2)(112,512)
//    val glmRE12: GeneralizedLinearModel = new LogisticRegressionModel(RE1Item2)
//
//    val glmRE1RDD = sc.parallelize(List(("RE1Item1", glmRE11), ("RE1Item2", glmRE12)))
//    val RE1Model = new RandomEffectModel(glmRE1RDD, "REModel1", "RE1Features")
//
//    // Random effect 2 has 3 items (of a different kind of model)
//    val numFeaturesRE2 = numFeaturesPerModel("RE2Features")
//    val RE2Item1 = CoefficientsTest.sparseCoefficients(numFeaturesRE2)(3,4,6)(321,421,621)
//    val glmRE21: GeneralizedLinearModel = new PoissonRegressionModel(RE2Item1)
//    val RE2Item2 = CoefficientsTest.sparseCoefficients(numFeaturesRE2)(4,5)(322,422)
//    val glmRE22: GeneralizedLinearModel = new PoissonRegressionModel(RE2Item2)
//    val RE2Item3 = CoefficientsTest.sparseCoefficients(numFeaturesRE2)(2,7,8)(323,423,523)
//    val glmRE23: GeneralizedLinearModel = new PoissonRegressionModel(RE2Item3)
//
//    val glmRE2RDD = sc.parallelize(List(("RE2Item1", glmRE21), ("RE2Item2", glmRE22), ("RE2Item3", glmRE23)))
//    val RE2Model = new RandomEffectModel(glmRE2RDD, "REModel2", "RE2Features")
//
//    // This GAME model has 1 fixed effect, and 2 different random effect models
//    GameModel(("fixed", FEModel), ("RE1", RE1Model), ("RE2", RE2Model))
//  }
//}
