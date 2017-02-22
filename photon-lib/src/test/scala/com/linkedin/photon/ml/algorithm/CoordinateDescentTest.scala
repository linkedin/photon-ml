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
package com.linkedin.photon.ml.algorithm

import org.apache.spark.rdd.RDD
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.{DataSet, GameDatum, KeyValueScore}
import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.model.{DatumScoringModel, GAMEModel}
import com.linkedin.photon.ml.optimization.OptimizationTracker
import com.linkedin.photon.ml.util.PhotonLogger

/**
 * Tests for [[CoordinateDescent]].
 */
class CoordinateDescentTest {

  abstract class MockDataSet extends DataSet[MockDataSet] {}
  type CoordinateType = Coordinate[MockDataSet]

  @DataProvider
  def coordinateCountProvider(): Array[Array[Integer]] = {
    (1 to 5).map(x => Array(Int.box(x))).toArray
  }

  /**
   * Tests for CoordinateDescent without validation.
   *
   * @param coordinateCount
   */
  @Test(dataProvider = "coordinateCountProvider")
  def testRun(coordinateCount: Int): Unit = {

    val numIterations = 10

    // Create Coordinate mocks
    val coordinateIds = (0 until coordinateCount).map("coordinate" + _)
    val coordinates: Seq[(String, CoordinateType)] =
      coordinateIds.map { coordinateId => (coordinateId, mock(classOf[CoordinateType])) }

    // Other mocks
    val evaluator = mock(classOf[Evaluator])
    val logger = mock(classOf[PhotonLogger])
    val gameModel = mock(classOf[GAMEModel])
    val tracker = mock(classOf[OptimizationTracker])
    val score = mock(classOf[KeyValueScore])
    val models = coordinates.map { _ =>
      mock(classOf[DatumScoringModel])
    }

    // KeyValueScore mock setup
    doReturn(score).when(score). +(score)
    doReturn(score).when(score).setName(Matchers.any(classOf[String]))
    doReturn(score).when(score).persistRDD(Matchers.any())

    // Per-coordinate mock setup
    (coordinates zip models).map { case ((coordinateId, coordinate), model) =>

      // Coordinate mock setup
      doReturn(score).when(coordinate).score(model)
      doReturn((model, Some(tracker))).when(coordinate).updateModel(model)
      doReturn((model, Some(tracker))).when(coordinate).updateModel(model, score)

      // GAMEModel mock setup
      doReturn(gameModel).when(gameModel).updateModel(coordinateId, model)
      doReturn(Some(model)).when(gameModel).getModel(coordinateId)
    }

    // Run coordinate descent - None = no validation
    val coordinateDescent =
      new CoordinateDescent(coordinates, evaluator, validatingDataAndEvaluatorsOption = None, logger)
    coordinateDescent.optimize(numIterations, gameModel)

    // Verify the calls to updateModel
    if (coordinates.length == 1) {
      verify(coordinates.head._2, times(numIterations)).updateModel(models.head)

    } else {
      (coordinates zip models).map { case ((coordinateId, coordinate), model) =>
        verify(coordinate, times(numIterations)).updateModel(model, score)
      }
    }
  }

  /**
   * Test CoordinateDescent with validation.
   */
  @Test
  def testBestModel(): Unit = {

    val (evaluatorCount, iterationCount) = (2, 3)

    val coordinate = mock(classOf[CoordinateType])
    val coordinates: Seq[(String, CoordinateType)] = List(("Coordinate0", coordinate))

    val lossEvaluator = mock(classOf[Evaluator])
    val logger = mock(classOf[PhotonLogger])
    val tracker = mock(classOf[OptimizationTracker])
    val (score, validationScore) = (mock(classOf[KeyValueScore]), mock(classOf[KeyValueScore]))
    val coordinateModel = mock(classOf[DatumScoringModel])
    val modelScores = mock(classOf[RDD[(Long, Double)]])
    val validationData = mock(classOf[RDD[(Long, GameDatum)]])

    val validationEvaluators = (0 until evaluatorCount).map { case (id) =>
      val mockEvaluator = mock(classOf[Evaluator])
      when(mockEvaluator.getEvaluatorName).thenReturn(s"validation evaluator $id")
      when(mockEvaluator.evaluate(modelScores)).thenReturn(id.toDouble)
      mockEvaluator
    }

    when(score + score).thenReturn(score)
    when(score.setName(Matchers.any())).thenReturn(score)
    when(score.persistRDD(Matchers.any())).thenReturn(score)

    when(validationScore + validationScore).thenReturn(validationScore)
    when(validationScore.scores).thenReturn(modelScores)
    when(validationScore.setName(Matchers.any())).thenReturn(validationScore)
    when(validationScore.persistRDD(Matchers.any())).thenReturn(validationScore)
    when(validationScore.materialize()).thenReturn(validationScore)

    // The very first GAME model will give rise to GAME model #1 via update,
    // and GAME model #1 will be the first one to go through best model selection
    val gameModels = (0 to iterationCount + 1).map { _ => mock(classOf[GAMEModel]) }
    (0 until iterationCount).map { i =>
      when(gameModels(i).getModel(Matchers.any())).thenReturn(Some(coordinateModel))
      when(gameModels(i).updateModel(Matchers.any(), Matchers.any())).thenReturn(gameModels(i + 1))
    }

    when(coordinateModel.score(Matchers.any())).thenReturn(validationScore)
    when(coordinate.score(Matchers.any())).thenReturn(score)
    when(coordinate.updateModel(coordinateModel)).thenReturn((coordinateModel, Some(tracker)))
    when(coordinate.updateModel(coordinateModel, score)).thenReturn((coordinateModel, Some(tracker)))

    // The evaluators are called iterationCount - 1 times, since on the first iteration
    // there is nothing to compare the model against
    val evaluator1 = validationEvaluators.head
    val evaluator2 = validationEvaluators(1)
    when(evaluator1.betterThan(Matchers.any(), Matchers.any()))
      .thenReturn(true)
      .thenReturn(false)
    when(evaluator2.betterThan(Matchers.any(), Matchers.any()))
      .thenReturn(false)
      .thenReturn(false)

    // Run coordinate descent
    val validationDataAndEvaluators
      = if (validationEvaluators.isEmpty) None else Option(validationData, validationEvaluators)
    val coordinateDescent = new CoordinateDescent(coordinates, lossEvaluator, validationDataAndEvaluators, logger)
    val (returnedModel, _) = coordinateDescent.optimize(iterationCount, gameModels.head)

    assert(returnedModel.hashCode == gameModels(2).hashCode())

    // Verify the calls to updateModel
    if (coordinates.length == 1) {
      verify(coordinates.head._2, times(iterationCount)).updateModel(coordinateModel)
    } else {
      verify(coordinate, times(iterationCount)).updateModel(coordinateModel, score)
    }

    // Verify the calls to the validation evaluator(s), if any
    validationDataAndEvaluators.map { case (data, evaluators) =>
      evaluators.map { case (evaluator) =>
        verify(evaluator, times(iterationCount)).evaluate(modelScores)
      }
    }
  }
}
