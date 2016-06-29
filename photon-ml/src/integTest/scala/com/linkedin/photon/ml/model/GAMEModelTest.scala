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

import org.apache.spark.SparkContext
import org.mockito.Mockito._
import org.testng.annotations.Test
import org.testng.Assert._

import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.optimization.LogisticRegressionOptimizationProblem
import com.linkedin.photon.ml.test.SparkTestUtils

class GAMEModelTest extends SparkTestUtils {

  import GAMEModelTest._

  @Test
  def testGetModel(): Unit = sparkTest("testGetModel") {

    val fixEffectModelName1 = "fix1"
    val randomEffectModelName1 = "random1"
    val fixEffectModelName2 = "fix2"
    val randomEffectModelName2 = "random2"

    val fixedEffectModel1 = getFixedEffectModel(sc, 1)
    val fixedEffectModel2 = getFixedEffectModel(sc, 2)
    val randomEffectModel1 = getRandomEffectModel(sc, 1)
    val randomEffectModel2 = getRandomEffectModel(sc, 2)

    // case 1: fixed effect model only
    val fixedEffectModelOnly =
      new GAMEModel(Map(fixEffectModelName1 -> fixedEffectModel1, fixEffectModelName2 -> fixedEffectModel2))
    assertEquals(fixedEffectModel1, fixedEffectModelOnly.getModel(fixEffectModelName1).get)
    assertEquals(fixedEffectModel2, fixedEffectModelOnly.getModel(fixEffectModelName2).get)
    assertTrue(fixedEffectModelOnly.getModel(randomEffectModelName1).isEmpty)

    // case 2: random effect model only
    val randomEffectModelOnly =
      new GAMEModel(Map(randomEffectModelName1 -> randomEffectModel1, randomEffectModelName2 -> randomEffectModel2))
    assertEquals(randomEffectModel1, randomEffectModelOnly.getModel(randomEffectModelName1).get)
    assertEquals(randomEffectModel2, randomEffectModelOnly.getModel(randomEffectModelName2).get)
    assertTrue(randomEffectModelOnly.getModel(fixEffectModelName2).isEmpty)

    // case 3: fixed and random effect model
    val fixedAndRandomEffectModel =
      new GAMEModel(Map(fixEffectModelName1 -> fixedEffectModel1, randomEffectModelName2 -> randomEffectModel2))
    assertEquals(fixedEffectModel1, fixedAndRandomEffectModel.getModel(fixEffectModelName1).get)
    assertEquals(randomEffectModel2, fixedAndRandomEffectModel.getModel(randomEffectModelName2).get)
    assertTrue(fixedAndRandomEffectModel.getModel(fixEffectModelName2).isEmpty)
    assertTrue(fixedAndRandomEffectModel.getModel(randomEffectModelName1).isEmpty)
  }

  @Test
  def testUpdateModelOfSameType(): Unit = sparkTest("testUpdateModelOfSameType") {

    val fixEffectModelName = "fix"
    val randomEffectModelName = "random"

    val fixedEffectModel1 = getFixedEffectModel(sc, 1)
    val fixedEffectModel2 = getFixedEffectModel(sc, 2)
    val randomEffectModel1 = getRandomEffectModel(sc, 1)
    val randomEffectModel2 = getRandomEffectModel(sc, 2)

    val gameModel11 =
      new GAMEModel(Map(fixEffectModelName -> fixedEffectModel1, randomEffectModelName -> randomEffectModel1))
    assertEquals(gameModel11.getModel(fixEffectModelName).get, fixedEffectModel1)
    assertEquals(gameModel11.getModel(randomEffectModelName).get, randomEffectModel1)
    val gameModel21 = gameModel11.updateModel(fixEffectModelName, fixedEffectModel2)
    assertEquals(gameModel21.getModel(fixEffectModelName).get, fixedEffectModel2)
    val gameModel22 = gameModel21.updateModel(randomEffectModelName, randomEffectModel2)
    assertEquals(gameModel22.getModel(randomEffectModelName).get, randomEffectModel2)
  }

  @Test(expectedExceptions = Array(classOf[UnsupportedOperationException]))
  def testUpdateModelOfDifferentType(): Unit = sparkTest("testUpdateModelOfDifferentType") {
    val fixEffectModelName = "fix"

    val fixedEffectModel = getFixedEffectModel(sc, 1)
    val randomEffectModel = getRandomEffectModel(sc, 1)

    val gameModel = new GAMEModel(Map(fixEffectModelName -> fixedEffectModel))
    gameModel.updateModel(fixEffectModelName, randomEffectModel)
  }

  @Test
  def testToMap(): Unit = sparkTest("testToMap") {
    val fixEffectModelName = "fix"
    val randomEffectModelName = "random"

    val fixedEffectModel = getFixedEffectModel(sc, 1)
    val randomEffectModel = getRandomEffectModel(sc, 1)

    val modelsMap = Map(fixEffectModelName -> fixedEffectModel, randomEffectModelName -> randomEffectModel)
    val gameModel = new GAMEModel(modelsMap)
    assertEquals(gameModel.toMap, modelsMap)
  }

  @Test
  def testPersistAndUnpersist(): Unit = sparkTest("testPersistAndUnpersist") {
    val randomEffectModelName = "random"
    val randomEffectModel = getRandomEffectModel(sc, 1)
    val modelsMap = Map(randomEffectModelName -> randomEffectModel)
    val gameModel = new GAMEModel(modelsMap)
    assertFalse(randomEffectModel.modelsRDD.getStorageLevel.isValid)
    gameModel.persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    assertEquals(randomEffectModel.modelsRDD.getStorageLevel, StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    gameModel.unpersist
    assertFalse(randomEffectModel.modelsRDD.getStorageLevel.isValid)
  }

  @Test
  def testEquals(): Unit = sparkTest("testPersistAndUnpersist") {
    val fixEffectModelName1 = "fix1"
    val randomEffectModelName1 = "random1"
    val fixEffectModelName2 = "fix2"
    val randomEffectModelName2 = "random2"

    val fixedEffectModel1 = getFixedEffectModel(sc, 1)
    val fixedEffectModel2 = getFixedEffectModel(sc, 2)
    val randomEffectModel1 = getRandomEffectModel(sc, 1)
    val randomEffectModel2 = getRandomEffectModel(sc, 1)

    val gameModel1111 =
      new GAMEModel(Map(fixEffectModelName1 -> fixedEffectModel1, randomEffectModelName1 -> randomEffectModel1))
    val gameModel1112 =
      new GAMEModel(Map(fixEffectModelName1 -> fixedEffectModel1, randomEffectModelName1 -> randomEffectModel2))
    val gameModel1212 =
      new GAMEModel(Map(fixEffectModelName1 -> fixedEffectModel2, randomEffectModelName1 -> randomEffectModel2))
    val gameModel1122 =
      new GAMEModel(Map(fixEffectModelName1 -> fixedEffectModel1, randomEffectModelName2 -> randomEffectModel2))
    val gameModel2121 =
      new GAMEModel(Map(fixEffectModelName2 -> fixedEffectModel1, randomEffectModelName2 -> randomEffectModel1))
    val gameModel2211 =
      new GAMEModel(Map(fixEffectModelName2 -> fixedEffectModel2, randomEffectModelName1 -> randomEffectModel1))
    val gameModel2212 =
      new GAMEModel(Map(fixEffectModelName2 -> fixedEffectModel2, randomEffectModelName1 -> randomEffectModel2))

    // Same name and model
    assertEquals(gameModel1111, gameModel1111)
    assertEquals(gameModel1111, gameModel1112)
    assertEquals(gameModel2211, gameModel2212)

    // Either name or model is different
    assertNotEquals(gameModel1212, gameModel1122)
    assertNotEquals(gameModel2121, gameModel2211)
    assertNotEquals(gameModel1212, gameModel2212)
  }
}

object GAMEModelTest {

  /**
    * Generate a toy fixed effect model
    *
    * @param sc the Spark context
    * @param coefficientDimension the dimension of the coefficients
    * @return a fixed effect model
    */
  protected def getFixedEffectModel(sc: SparkContext, coefficientDimension: Int): FixedEffectModel = {
    // Coefficients parameter
    val glm: GeneralizedLinearModel = LogisticRegressionOptimizationProblem.initializeZeroModel(coefficientDimension)

    // Meta data
    val featureShardId = "featureShardId"

    // Fixed effect model
    new FixedEffectModel(sc.broadcast(glm), featureShardId)
  }

  /**
    * Generate a toy random effect model
    *
    * @param sc the Spark context
    * @param coefficientDimension the dimension of the coefficients
    * @return a random effect model
    */
  protected def getRandomEffectModel(sc: SparkContext, coefficientDimension: Int): RandomEffectModel = {
    // Coefficients parameter
    val glm: GeneralizedLinearModel = LogisticRegressionOptimizationProblem.initializeZeroModel(coefficientDimension)

    // Meta data
    val featureShardId = "featureShardId"
    val randomEffectId = "randomEffectId"

    // Random effect model
    val numCoefficients = 5
    val modelsRDD = sc.parallelize(Seq.tabulate(numCoefficients)(i => (i.toString, glm)))
    new RandomEffectModel(modelsRDD, randomEffectId, featureShardId)
  }
}
