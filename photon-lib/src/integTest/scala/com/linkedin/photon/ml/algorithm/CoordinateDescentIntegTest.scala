/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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

import scala.util.Random

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.evaluation.{EvaluationResults, EvaluationSuite, Evaluator, EvaluatorType}
import com.linkedin.photon.ml.model.DatumScoringModel
import com.linkedin.photon.ml.optimization.OptimizationTracker
import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.PhotonLogger

/**
 * Unit tests for [[CoordinateDescent]].
 */
class CoordinateDescentIntegTest extends SparkTestUtils {

  import CoordinateDescentIntegTest._

  abstract class MockDataset extends Dataset[MockDataset] {}
  abstract class MockDatumScoringModel extends DatumScoringModel with RDDLike {}
  type CoordinateType = Coordinate[MockDataset]

  @DataProvider
  def coordinateCountProvider(): Array[Array[Integer]] = (1 to 5).map(x => Array(Int.box(x))).toArray

  /**
   * Tests for [[CoordinateDescent]] without validation.
   *
   * @param coordinateCount Number of coordinates to generate for test
   */
  @Test(dataProvider = "coordinateCountProvider")
  def testRun(coordinateCount: Int): Unit = sparkTest("testRun") {

    val numIterations = 10

    //
    // Create mocks
    //

    val coordinateIds = (0 until coordinateCount).map("coordinate" + _)
    val coordinatesAndModels = coordinateIds.map { coordinateId =>
      (coordinateId, mock(classOf[CoordinateType]), mock(classOf[MockDatumScoringModel]))
    }
    val tracker = mock(classOf[OptimizationTracker])
    val score = mock(classOf[CoordinateDataScores])

    //
    // Mock setup
    //

    // Scores mock setup
    when(score + score).thenReturn(score)
    when(score - score).thenReturn(score)

    doReturn(score).when(score).+(Matchers.any(classOf[CoordinateDataScores]))
    doReturn(score).when(score).setName(Matchers.any(classOf[String]))
    doReturn(score).when(score).persistRDD(Matchers.any(classOf[StorageLevel]))
    doReturn(score).when(score).materialize()
    doReturn(score).when(score).unpersistRDD()

    // Per-coordinate mock setup
    coordinatesAndModels.foreach { case (_, coordinate, model) =>

      // Coordinate mock setup
      doReturn(score).when(coordinate).score(model)
      doReturn((model, Some(tracker))).when(coordinate).trainModel(model)
      doReturn((model, Some(tracker))).when(coordinate).trainModel(model, score)

      doReturn(model).when(model).setName(Matchers.any(classOf[String]))
      doReturn(model).when(model).persistRDD(Matchers.any(classOf[StorageLevel]))
      doReturn(model).when(model).materialize()
      doReturn(model).when(model).unpersistRDD()
    }

    //
    // Run coordinate descent
    //

    val coordinates = coordinatesAndModels
      .map { case (coordinateId, coordinate, _) =>
        (coordinateId, coordinate)
      }
      .toMap
    val initialModels = coordinatesAndModels
      .map { case (coordinateId, _, model) =>
        (coordinateId, model)
      }
      .toMap

    val coordinateDescent = new CoordinateDescent(
      coordinateIds,
      numIterations,
      validationDataAndEvaluationSuiteOpt = None,
      lockedCoordinates = Set(),
      MOCK_LOGGER)
    coordinateDescent.run(coordinates, Some(initialModels))

    //
    // Verify results
    //

    val (_, firstCoordinate, firstModel) = coordinatesAndModels.head
    verify(firstCoordinate, times( 1)).trainModel(firstModel)
    if (coordinateCount == 1) {
      verify(firstCoordinate, never()).trainModel(firstModel, score)
    } else {
      verify(firstCoordinate, times(numIterations - 1)).trainModel(firstModel, score)
    }

    coordinatesAndModels.tail.foreach { case (_, coordinate, model) =>
      verify(coordinate, times(numIterations)).trainModel(model, score)
    }
  }

  /**
   * Test [[CoordinateDescent]] with validation.
   */
  @Test
  def testBestModel(): Unit = sparkTest("testBestModel") {

    val coordinateCount = 5

    val evaluatorType = EvaluatorType.RMSE
    val evaluations = (0 until coordinateCount).map(_ => Random.nextDouble())

    val bestEvaluation = evaluations.min
    val evaluationResults = evaluations.map { evaluation =>
      EvaluationResults(Map(evaluatorType -> evaluation), evaluatorType)
    }
    val evaluationResultsIterator = evaluationResults.iterator

    //
    // Create mocks
    //

    val coordinateIds = (0 until coordinateCount).map(s"coordinate" + _)
    val coordinates = coordinateIds.map { coordinateId =>
      (coordinateId, mock(classOf[CoordinateType]))
    }
    val tracker = mock(classOf[OptimizationTracker])
    val trainingScore = mock(classOf[CoordinateDataScores])
    val validationData = mock(classOf[RDD[(UniqueSampleId, GameDatum)]])
    val validationScoresList = coordinateIds.map(_ =>  mock(classOf[CoordinateDataScores]))
    val validationScoresIterator = validationScoresList.iterator
    val evaluationSuite = mock(classOf[EvaluationSuite])
    val evaluator = mock(classOf[Evaluator])

    //
    // Mock setup
    //

    // Training scores mock setup
    when(trainingScore + trainingScore).thenReturn(trainingScore)

    doReturn(trainingScore).when(trainingScore).+(Matchers.any(classOf[CoordinateDataScores]))
    doReturn(trainingScore).when(trainingScore).setName(Matchers.any(classOf[String]))
    doReturn(trainingScore).when(trainingScore).persistRDD(Matchers.any(classOf[StorageLevel]))
    doReturn(trainingScore).when(trainingScore).materialize()
    doReturn(trainingScore).when(trainingScore).unpersistRDD()

    // Evaluation suite mock setup
    doReturn(evaluator).when(evaluationSuite).primaryEvaluator
    doReturn(evaluatorType).when(evaluator).evaluatorType

    // Per-coordinate mock setup
    coordinates.foreach { case (_, coordinate) =>

      val model = mock(classOf[MockDatumScoringModel])
      val validationScores = validationScoresIterator.next()
      val rawValidationScores = mock(classOf[RDD[(UniqueSampleId, Double)]])

      // Coordinate mock setup
      doReturn(trainingScore).when(coordinate).score(model)
      doReturn((model, Some(tracker))).when(coordinate).trainModel()
      doReturn((model, Some(tracker))).when(coordinate).trainModel(trainingScore)

      // Model mock setup
      doReturn(model).when(model).setName(Matchers.any(classOf[String]))
      doReturn(model).when(model).persistRDD(Matchers.any(classOf[StorageLevel]))
      doReturn(model).when(model).materialize()
      doReturn(model).when(model).unpersistRDD()
      doReturn(validationScores).when(model).scoreForCoordinateDescent(validationData)

      // Evaluation results mock setup
      doReturn(rawValidationScores).when(validationScores).scores
      doReturn(evaluationResultsIterator.next()).when(evaluationSuite).evaluate(rawValidationScores)
    }

    // Validation scores mock setup:
    // Don't know the order the mocks will be summed in (due to storage in Map object), but always want the latest mock
    // to be selected
    for (i <- 0 until coordinateCount) {
      for (j <- (i + 1) until coordinateCount) {
        doReturn(validationScoresList(j)).when(validationScoresList(i)).+(validationScoresList(j))
        doReturn(validationScoresList(j)).when(validationScoresList(j)).+(validationScoresList(i))
      }
    }

    validationScoresList.reduce { (validationScores1: CoordinateDataScores, validationScores2: CoordinateDataScores) =>
      doReturn(validationScores2).when(validationScores2).+(validationScores1)

      validationScores2
    }

    //
    // Run coordinate descent
    //

    val coordinateDescent = new CoordinateDescent(
      coordinateIds,
      1,
      Some((validationData, evaluationSuite)),
      lockedCoordinates = Set(),
      MOCK_LOGGER)
    val (_, result) = coordinateDescent.run(coordinates.toMap, None)

    //
    // Verify results
    //

    assertEquals(result.get.evaluations(evaluatorType), bestEvaluation, MathConst.EPSILON)
  }
}

object CoordinateDescentIntegTest {

  val MOCK_LOGGER: PhotonLogger = mock(classOf[PhotonLogger])
}
