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
package com.linkedin.photon.ml.algorithm

import breeze.linalg.SparseVector
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.RandomEffectDataset
import com.linkedin.photon.ml.function.glm.SingleNodeGLMLossFunction
import com.linkedin.photon.ml.model.{Coefficients, RandomEffectModel}
import com.linkedin.photon.ml.optimization.game.RandomEffectOptimizationProblem
import com.linkedin.photon.ml.projector.{LinearSubspaceProjector, LinearSubspaceProjectorTest}
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.{GameTestUtils, MathUtils}

/**
 * Integration tests for [[RandomEffectCoordinate]].
 */
class RandomEffectCoordinateIntegTest extends SparkTestUtils with GameTestUtils {

  import RandomEffectCoordinateIntegTest._


  /**
   *
   */
  @Test
  def testProjectModelForward(): Unit = sparkTest("testProjectForward") {

    // Feature vectors for Coefficients
    val fV1 = new SparseVector[Double](Array(0, 1, 2, 3), Array(1D, 1D, 2D, 3D), ORIGINAL_SPACE_DIMENSION)
    val fV2 = new SparseVector[Double](Array(4, 5, 6, 7), Array(4D, 5D, 6D, 7D), ORIGINAL_SPACE_DIMENSION)

    // Coefficients for GeneralizedLinearModels
    val coef1 = Coefficients(fV1)
    val coef2 = Coefficients(fV2)

    // Points for LocalDataSet
    val gLM1 = LogisticRegressionModel(coef1)
    val gLM2 = LogisticRegressionModel(coef2)

    // Create LinearSubspaceREDProjector
    val linearSubspaceProjectors = sc.parallelize(Seq((REID1, PROJECTOR_1), (REID2, PROJECTOR_2)))
    // Create RandomEffectDataset
    val randomEffectModel = new RandomEffectModel(
      sc.parallelize(Seq((REID1, gLM1), (REID2, gLM2))),
      "someType",
      "someShard")

    val mockRandomEffectDataset = mock(classOf[RandomEffectDataset])
    val mockRandomEffectOptimizationProblem = mock(classOf[RandomEffectOptimizationProblem[SingleNodeGLMLossFunction]])

    doReturn(linearSubspaceProjectors).when(mockRandomEffectDataset).projectors

    val coordinate = new RandomEffectCoordinate(mockRandomEffectDataset, mockRandomEffectOptimizationProblem)

    // Expected values
    val expectedGLM1ThroughProjector1 = Set[(Int, Double)](
      (PROJECTOR_1.originalToProjectedSpaceMap(1), 1D),
      (PROJECTOR_1.originalToProjectedSpaceMap(3), 3D))
    val expectedGLM2ThroughProjector2 = Set[(Int, Double)](
      (PROJECTOR_2.originalToProjectedSpaceMap(4), 4D),
      (PROJECTOR_2.originalToProjectedSpaceMap(6), 6D))

    // Do projection
    val projectedRandomEffectModel = coordinate.projectModelForward(randomEffectModel)

    // Check that feature vectors were properly projected
    val projectedGLMs = projectedRandomEffectModel.modelsRDD.take(2).toMap
    val projectedGLM1 = projectedGLMs(REID1)
    val projectedGLM2 = projectedGLMs(REID2)

    val projectedFV1 = projectedGLM1.coefficients.means
    val projectedFV2 = projectedGLM2.coefficients.means

    assertEquals(LinearSubspaceProjectorTest.getActiveTuples(projectedFV1), expectedGLM1ThroughProjector1)
    assertEquals(LinearSubspaceProjectorTest.getActiveTuples(projectedFV2), expectedGLM2ThroughProjector2)
  }

  /**
   *
   */
  @Test
  def testProjectModelBackward(): Unit = sparkTest("testProjectForward") {

    // Feature vectors for Coefficients
    val fV1 = new SparseVector[Double](Array(0, 1), Array(1D, 1D), PROJECTED_SPACE_DIMENSION)
    val fV2 = new SparseVector[Double](Array(2, 3), Array(1D, 1D), PROJECTED_SPACE_DIMENSION)

    // Coefficients for GeneralizedLinearModels
    val coef1 = Coefficients(fV1)
    val coef2 = Coefficients(fV2)

    // Points for LocalDataSet
    val gLM1 = LogisticRegressionModel(coef1)
    val gLM2 = LogisticRegressionModel(coef2)

    // Create LinearSubspaceREDProjector
    val linearSubspaceProjectors = sc.parallelize(Seq((REID1, PROJECTOR_1), (REID2, PROJECTOR_2)))
    // Create RandomEffectDataset
    val randomEffectModel = new RandomEffectModel(
      sc.parallelize(Seq((REID1, gLM1), (REID2, gLM2))),
      "someType",
      "someShard")

    val mockRandomEffectDataset = mock(classOf[RandomEffectDataset])
    val mockRandomEffectOptimizationProblem = mock(classOf[RandomEffectOptimizationProblem[SingleNodeGLMLossFunction]])

    doReturn(linearSubspaceProjectors).when(mockRandomEffectDataset).projectors

    val coordinate = new RandomEffectCoordinate(mockRandomEffectDataset, mockRandomEffectOptimizationProblem)

    // Expected values
    val expectedGLM1ThroughProjector1 = Set[(Int, Double)](
      (PROJECTOR_1.projectedToOriginalSpaceMap(0), 1D),
      (PROJECTOR_1.projectedToOriginalSpaceMap(1), 1D))
    val expectedGLM2ThroughProjector2 = Set[(Int, Double)](
      (PROJECTOR_2.projectedToOriginalSpaceMap(2), 1D),
      (PROJECTOR_2.projectedToOriginalSpaceMap(3), 1D))

    // Do projection
    val projectedRandomEffectModel = coordinate.projectModelBackward(randomEffectModel)

    // Check that feature vectors were properly projected
    val projectedGLMs = projectedRandomEffectModel.modelsRDD.take(2).toMap
    val projectedGLM1 = projectedGLMs(REID1)
    val projectedGLM2 = projectedGLMs(REID2)

    val projectedFV1 = projectedGLM1.coefficients.means
    val projectedFV2 = projectedGLM2.coefficients.means

    val projectedFV1ActiveTuples = LinearSubspaceProjectorTest.getActiveTuples(projectedFV1)
    val projectedFV2ActiveTuples = LinearSubspaceProjectorTest.getActiveTuples(projectedFV2)

    assertEquals(projectedFV1ActiveTuples, expectedGLM1ThroughProjector1)
    assertEquals(projectedFV2ActiveTuples, expectedGLM2ThroughProjector2)
    assertTrue(projectedFV1ActiveTuples.intersect(projectedFV2ActiveTuples).isEmpty)
  }

  @DataProvider
  def numEntitiesDataProvider(): Array[Array[Any]] = Array(Array(1), Array(2), Array(10))

  /**
   * Test that a [[RandomEffectCoordinate]] can train a new model.
   *
   * @param numEntities The number of unique per-entity models to train
   */
  @Test(dataProvider = "numEntitiesDataProvider", dependsOnMethods = Array("testScore"))
  def testTrainModel(numEntities: Int): Unit = sparkTest("testUpdateModel") {

    val (coordinate, model) = generateRandomEffectCoordinateAndModel(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_ID,
      numEntities,
      NUM_TRAINING_SAMPLES,
      DIMENSIONALITY)

    // Score before model update
    val score = coordinate.score(model)
    assertTrue(score.scoresRdd.map(_._2).collect.forall(MathUtils.isAlmostZero))

    // Train models
    val (newModelWithoutInitial, _) = coordinate.trainModel()
    val (newModelWithInitial, _) = coordinate.trainModel(model)

    assertNotEquals(newModelWithoutInitial, model)
    assertNotEquals(newModelWithInitial, model)

    // Score after model update
    val newScoreWithoutInitial = coordinate.score(newModelWithoutInitial)
    val newScoreWithInitial = coordinate.score(newModelWithInitial)

    assertFalse(newScoreWithoutInitial.scoresRdd.map(_._2).collect.forall(MathUtils.isAlmostZero))
    assertFalse(newScoreWithInitial.scoresRdd.map(_._2).collect.forall(MathUtils.isAlmostZero))

    newScoreWithoutInitial
      .scoresRdd
      .join(newScoreWithInitial.scoresRdd)
      .values
      .foreach { case (score1, score2) =>
        assertEquals(score1, score2, MathConst.EPSILON)
      }
  }

  /**
   * Test that a [[RandomEffectCoordinate]] can train a new model, using an existing [[RandomEffectModel]] as a starting
   * point for optimization, and retain existing models for which new data does not exist.
   */
  @Test
  def testTrainWithExtraModel(): Unit = sparkTest("testUpdateInitialModel") {

    val extraREID = "reExtra"
    val (coordinate, baseModel) = generateRandomEffectCoordinateAndModel(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_ID,
      numEntities = 1,
      NUM_TRAINING_SAMPLES,
      DIMENSIONALITY)

    // Add in an item that exists in the prior model, but not the data
    val randomEffectIds = baseModel.modelsRDD.keys.collect() :+ extraREID
    val extraModel =
      baseModel.update(sc.parallelize(generateLinearModelsForRandomEffects(randomEffectIds, DIMENSIONALITY)))
    val (newModel, _) = coordinate.trainModel(extraModel)

    newModel match {
      case randomEffectModel: RandomEffectModel =>
        // Make sure that the prior model items are still there
        assertEquals(randomEffectModel.modelsRDD.map(_._1).collect.toSet, randomEffectIds.toSet)

      case other =>
        fail(s"Unexpected model type: ${other.getClass.getName}")
    }
  }

  /**
   * Test that a [[RandomEffectCoordinate]] can score data using a [[RandomEffectModel]].
   *
   * @param numEntities The number of unique per-entity models to train
   */
  @Test(dataProvider = "numEntitiesDataProvider")
  def testScore(numEntities: Int): Unit = sparkTest("testScore") {

    val (coordinate, model) = generateRandomEffectCoordinateAndModel(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_ID,
      numEntities,
      NUM_TRAINING_SAMPLES,
      DIMENSIONALITY)

    val score = coordinate.score(model)

    assertEquals(score.scoresRdd.count, numEntities * NUM_TRAINING_SAMPLES)
    assertTrue(score.scoresRdd.map(_._2).collect.forall(MathUtils.isAlmostZero))
  }
}

object RandomEffectCoordinateIntegTest {

  // IDs
  private val RANDOM_EFFECT_TYPE = "random-effect-1"
  private val FEATURE_SHARD_ID = "shard1"

  // RandomEffectDataset dimensions
  private val NUM_TRAINING_SAMPLES = 1000
  private val DIMENSIONALITY = 10

  // Projector dimensions
  private val ORIGINAL_SPACE_DIMENSION = 8
  private val PROJECTED_SPACE_DIMENSION = 4

  // Random effect IDs
  private val REID1 = "1"
  private val REID2 = "2"

  // Projector 1 filters odd numbers
  private val PROJECTOR_1 = new LinearSubspaceProjector(Set(1, 3, 5, 7), ORIGINAL_SPACE_DIMENSION)
  // Projector 2 filters even numbers
  private val PROJECTOR_2 = new LinearSubspaceProjector(Set(0, 2, 4, 6), ORIGINAL_SPACE_DIMENSION)
}
