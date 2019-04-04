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

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Types.{CoordinateId, UniqueSampleId}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.evaluation.{EvaluationResults, EvaluationSuite, EvaluatorType}
import com.linkedin.photon.ml.model.DatumScoringModel
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}
import com.linkedin.photon.ml.util.PhotonLogger

/**
 * Unit tests for [[CoordinateDescent]].
 */
class CoordinateDescentTest {

  import CoordinateDescentTest._

  type CoordinateType = Coordinate[MockDataset]

  @DataProvider
  def invalidInput: Array[Array[Any]] = {

    // Mock parameters
    val goodIter = 1
    val coordinateId1 = "someCoordinate"
    val coordinateId2 = "someOtherCoordinate"
    val goodUpdateSequence = Seq(coordinateId1)
    val badUpdateSequence = Seq(coordinateId1, coordinateId1)
    val goodLockedCoordinates = Set(coordinateId1)
    val badLockedCoordinates = Set(coordinateId2)

    Array(
      // Repeated coordinates in the update sequence
      Array(badUpdateSequence, goodIter, goodLockedCoordinates),
      // 0 iterations
      Array(goodUpdateSequence, 0, goodLockedCoordinates),
      // Negative iterations
      Array(goodUpdateSequence, -1, goodLockedCoordinates),
      // Locked coordinates missing from update sequence
      Array(goodUpdateSequence, goodIter, badLockedCoordinates),
      // All coordinates locked
      Array(goodUpdateSequence, goodIter, goodUpdateSequence.toSet))
  }

  /**
   * Test [[CoordinateDescent]] creation with invalid input.
   *
   * @param updateSequence The order in which to update coordinates
   * @param descentIterations Number of coordinate descent iterations (updates to each coordinate in order)
   * @param lockedCoordinates Set of locked coordinates within the initial model for performing partial retraining
   */
  @Test(dataProvider = "invalidInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testCheckInvariants(
      updateSequence: Seq[CoordinateId],
      descentIterations: Int,
      lockedCoordinates: Set[CoordinateId]): Unit =
    new CoordinateDescent(
      updateSequence,
      descentIterations,
      None,
      lockedCoordinates,
      MOCK_LOGGER)

  @DataProvider
  def invalidRunInput: Array[Array[Any]] = {

    // Mocks
    val datumScoringModel = mock(classOf[DatumScoringModel])
    val coordinate = mock(classOf[CoordinateType])

    // Mock parameters
    val updateSequence = Seq("a", "b", "c", "d")
    val lockedCoordinates = Set("a", "c")
    val descentIterations = 1
    val goodCoordinates = Map("a" -> coordinate, "b" -> coordinate,"c" -> coordinate, "d" -> coordinate)
    val badCoordinates = Map("a" -> coordinate, "b" -> coordinate, "d" -> coordinate)
    val goodInitialModels = Some(lockedCoordinates.map((_, datumScoringModel)).toMap)
    val badInitialModels = Some(Map[CoordinateId, DatumScoringModel]("a" -> datumScoringModel))

    val coordinateDescent = new CoordinateDescent(
      updateSequence,
      descentIterations,
      validationDataAndEvaluationSuiteOpt = None,
      lockedCoordinates,
      MOCK_LOGGER)

    Array(
      // Coordinates to train are missing from the coordinates map
      Array(coordinateDescent, badCoordinates, goodInitialModels),
      // No initial models provided
      Array(coordinateDescent, goodCoordinates, None),
      // Locked coordinate without initial model
      Array(coordinateDescent, goodCoordinates, badInitialModels))
  }

  /**
   * Test attempts to run coordinate descent with invalid input.
   *
   * @param coordinateDescent A pre-built [[CoordinateDescent]] object to attempt to run
   * @param coordinates A map of optimization problem coordinates (optimization sub-problems)
   * @param initialModelsOpt
   */
  @Test(dataProvider = "invalidRunInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidRun(
      coordinateDescent: CoordinateDescent,
      coordinates: Map[CoordinateId, CoordinateType],
      initialModelsOpt: Option[Map[CoordinateId, DatumScoringModel]]): Unit =
    coordinateDescent.run(coordinates, initialModelsOpt)

  @DataProvider
  def trainCoordinateModelInput(): Array[Array[Any]] = {

    val mockCoordinate = mock(classOf[CoordinateType])
    val mockInitialModel = mock(classOf[DatumScoringModel])
    val mockResiduals = mock(classOf[CoordinateDataScores])

    val mockNewModel1 = mock(classOf[DatumScoringModel])
    val mockNewModel2 = mock(classOf[DatumScoringModel])
    val mockNewModel3 = mock(classOf[DatumScoringModel])
    val mockNewModel4 = mock(classOf[DatumScoringModel])

    doReturn((mockNewModel1, None)).when(mockCoordinate).trainModel(mockInitialModel, mockResiduals)
    doReturn((mockNewModel2, None)).when(mockCoordinate).trainModel(mockInitialModel)
    doReturn((mockNewModel3, None)).when(mockCoordinate).trainModel(mockResiduals)
    doReturn((mockNewModel4, None)).when(mockCoordinate).trainModel()

    Array(
      Array(mockCoordinate, Some(mockInitialModel), Some(mockResiduals), mockNewModel1),
      Array(mockCoordinate, Some(mockInitialModel), None, mockNewModel2),
      Array(mockCoordinate, None, Some(mockResiduals), mockNewModel3),
      Array(mockCoordinate, None, None, mockNewModel4))
  }

  /**
   * Test that a new [[DatumScoringModel]] can be trained correctly.
   *
   * @param coordinate The coordinate for which to train a new model
   * @param initialModelOpt An optional initial model from whose coefficients to begin optimization
   * @param residualsOpt Optional residual scores from previous coordinates
   * @param expectedResult Expected model to be returned based on input
   */
  @Test(dataProvider = "trainCoordinateModelInput")
  def testTrainCoordinateModel(
      coordinate: CoordinateType,
      initialModelOpt: Option[DatumScoringModel],
      residualsOpt: Option[CoordinateDataScores],
      expectedResult: DatumScoringModel): Unit = {

    val coordinateId = "mockCoordinateId"
    val iteration = 1

    val result = CoordinateDescent.trainCoordinateModel(
      coordinateId,
      coordinate,
      iteration,
      initialModelOpt,
      residualsOpt)(
      MOCK_LOGGER)

    assertTrue(result.eq(expectedResult))
  }

  /**
   * Test that locked coordinates will not have a new model trained.
   */
  @Test(dependsOnMethods = Array("testTrainCoordinateModel"))
  def testTrainOrFetchCoordinateModel(): Unit = {

    val mockCoordinate = mock(classOf[CoordinateType])
    val mockInitialModel = mock(classOf[DatumScoringModel])
    val mockNewModel = mock(classOf[DatumScoringModel])

    val trainingCoordinateId = "trainingCoordinateId"
    val lockedCoordinateId = "lockedCoordinateId"
    val coordinatesToTrain = Seq(trainingCoordinateId)
    val iteration = 1

    doReturn((mockNewModel, None)).when(mockCoordinate).trainModel(mockInitialModel)

    val newModel = CoordinateDescent.trainOrFetchCoordinateModel(
      trainingCoordinateId,
      mockCoordinate,
      coordinatesToTrain,
      iteration,
      Some(mockInitialModel),
      None)(
      MOCK_LOGGER)
    val lockedModel = CoordinateDescent.trainOrFetchCoordinateModel(
      lockedCoordinateId,
      mockCoordinate,
      coordinatesToTrain,
      iteration,
      Some(mockInitialModel),
      None)(
      MOCK_LOGGER)

    assertTrue(newModel.eq(mockNewModel))
    assertTrue(lockedModel.eq(mockInitialModel))
  }

  /**
   * Test that trained models are evaluated on validation data correctly.
   */
  @Test
  def testEvaluateModel(): Unit = {

    val mockModel = mock(classOf[DatumScoringModel])
    val mockValidationData = mock(classOf[RDD[(UniqueSampleId, GameDatum)]])
    val mockValidationScores = mock(classOf[CoordinateDataScores])
    val mockRawScores = mock(classOf[RDD[(UniqueSampleId, Double)]])
    val mockEvaluationSuite = mock(classOf[EvaluationSuite])
    val mockEvaluationResults = mock(classOf[EvaluationResults])

    val evaluatorType = EvaluatorType.AUC
    val evaluation = 1D
    val evaluations = Map(evaluatorType -> evaluation)

    doReturn(mockValidationScores).when(mockModel).scoreForCoordinateDescent(mockValidationData)
    doReturn(mockRawScores).when(mockValidationScores).scores
    doReturn(mockEvaluationResults).when(mockEvaluationSuite).evaluate(mockRawScores)
    doReturn(evaluations).when(mockEvaluationResults).evaluations

    val result = CoordinateDescent.evaluateModel(mockModel, mockValidationData, mockEvaluationSuite)(MOCK_LOGGER)

    assertTrue(result.eq(mockEvaluationResults))
  }

  /**
   * Test that [[RDDLike]] [[DatumScoringModel]] objects
   */
  @Test
  def testPersistModel(): Unit = {

    val mockBroadcastLike = mock(classOf[MockBroadcastModel])
    val mockRDDLike = mock(classOf[MockRDDModel])

    val coordinateId = "mockCoordinate"
    val iteration = 1

    doReturn(mockRDDLike).when(mockRDDLike).setName(Matchers.any(classOf[String]))
    doReturn(mockRDDLike).when(mockRDDLike).persistRDD(StorageLevel.DISK_ONLY)
    doReturn(mockRDDLike).when(mockRDDLike).materialize()

    CoordinateDescent.persistModel(mockBroadcastLike, coordinateId, iteration)
    CoordinateDescent.persistModel(mockRDDLike, coordinateId, iteration)

    verify(mockRDDLike, times(1)).setName(Matchers.any(classOf[String]))
    verify(mockRDDLike, times(1)).persistRDD(StorageLevel.DISK_ONLY)
    verify(mockRDDLike, times(1)).materialize()
  }

  /**
   * Test that [[CoordinateDataScores]] containing summed [[Coordinate]] scores are persisted correctly.
   */
  @Test
  def testPersistSummedScores(): Unit = {

    val mockScores = mock(classOf[CoordinateDataScores])

    doReturn(mockScores).when(mockScores).setName(Matchers.any(classOf[String]))
    doReturn(mockScores).when(mockScores).persistRDD(StorageLevel.MEMORY_AND_DISK_SER)
    doReturn(mockScores).when(mockScores).materialize()

    CoordinateDescent.persistSummedScores(mockScores)

    verify(mockScores, times(1)).setName(Matchers.any(classOf[String]))
    verify(mockScores, times(1)).persistRDD(StorageLevel.MEMORY_AND_DISK_SER)
    verify(mockScores, times(1)).materialize()
  }

  /**
   * Test that models are unpersisted correctly.
   */
  @Test
  def testUnpersistModel(): Unit = {

    val mockBroadcastLike = mock(classOf[MockBroadcastModel])
    val mockRDDLike = mock(classOf[MockRDDModel])

    doReturn(mockBroadcastLike).when(mockBroadcastLike).unpersistBroadcast()
    doReturn(mockRDDLike).when(mockRDDLike).unpersistRDD()

    CoordinateDescent.unpersistModel(mockBroadcastLike.asInstanceOf[DatumScoringModel])
    CoordinateDescent.unpersistModel(mockRDDLike.asInstanceOf[DatumScoringModel])

    verify(mockBroadcastLike, times(1)).unpersistBroadcast()
    verify(mockRDDLike, times(1)).unpersistRDD()
  }
}

object CoordinateDescentTest {

  abstract class MockDataset extends Dataset[MockDataset]
  abstract class MockBroadcastModel extends DatumScoringModel with BroadcastLike
  abstract class MockRDDModel extends DatumScoringModel with RDDLike

  val MOCK_LOGGER: PhotonLogger = mock(classOf[PhotonLogger])
}
