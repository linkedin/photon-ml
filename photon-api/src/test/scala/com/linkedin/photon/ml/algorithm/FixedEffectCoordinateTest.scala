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

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.data.{FixedEffectDataset, LabeledPoint}
import com.linkedin.photon.ml.function.DistributedObjectiveFunction
import com.linkedin.photon.ml.model.FixedEffectModel
import com.linkedin.photon.ml.optimization.{DistributedOptimizationProblem, OptimizationStatesTracker}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Unit tests for [[FixedEffectCoordinate]].
 */
class FixedEffectCoordinateTest {

  import FixedEffectCoordinateTest._

  /**
   * Test that a [[FixedEffectCoordinate]] can be updated with a new [[FixedEffectDataset]].
   */
  @Test
  def testUpdateCoordinateWithDataset(): Unit = {

    val dataset = mock(classOf[FixedEffectDataset])
    val newDataset = mock(classOf[FixedEffectDataset])
    val optimizationProblem = mock(classOf[DistributedOptimizationProblem[DistributedObjectiveFunction]])

    val coordinate = new MockFixedEffectCoordinate(dataset, optimizationProblem)
    val newCoordinate = coordinate.updateCoordinateWithDataset(newDataset)

    assertFalse(newCoordinate.publicDataset.eq(dataset))
    assertTrue(newCoordinate.publicDataset.eq(newDataset))
    assertTrue(newCoordinate.publicOptimizationProblem.eq(optimizationProblem))
  }

  @DataProvider
  def trainModelInput: Array[Array[Any]] = {

    val rawModel = mock(classOf[GeneralizedLinearModel])
    val fixedEffectModel = mock(classOf[FixedEffectModel])

    doReturn(rawModel).when(fixedEffectModel).model

    Array(
      Array[Any](Some(fixedEffectModel)),
      Array[Any](None))
  }

  /**
   * Test that a [[FixedEffectCoordinate]] can train a new (mocked) model.
   */
  @Test(dataProvider = "trainModelInput")
  def testTrainModel(initialModelOpt: Option[FixedEffectModel]): Unit = {

    // Create mocks
    val dataset = mock(classOf[FixedEffectDataset])
    val optimizationProblem = mock(classOf[DistributedOptimizationProblem[DistributedObjectiveFunction]])
    val updatedModel = mock(classOf[GeneralizedLinearModel])
    val labeledPoints = mock(classOf[RDD[(UniqueSampleId, LabeledPoint)]])
    val sparkContext = mock(classOf[SparkContext])
    val modelBroadcast = mock(classOf[Broadcast[GeneralizedLinearModel]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])

    val featureShardId = "mockShardId"

    // Dataset
    doReturn(labeledPoints).when(dataset).labeledPoints
    doReturn(featureShardId).when(dataset).featureShardId

    // Optimization problem
    doReturn(Some(statesTracker)).when(optimizationProblem).getStatesTracker

    // New model
    doReturn(TaskType.LOGISTIC_REGRESSION).when(updatedModel).modelType

    // RDD
    doReturn(sparkContext).when(labeledPoints).sparkContext

    // Spark context
    doReturn(modelBroadcast).when(sparkContext).broadcast(updatedModel)

    // Broadcast model
    doReturn(updatedModel).when(modelBroadcast).value

    val coordinate = new FixedEffectCoordinate(dataset, optimizationProblem)
    val (newModel, _) = initialModelOpt match {

      case Some(initialModel) =>
        val rawModel = initialModel.model

        doReturn(updatedModel).when(optimizationProblem).runWithSampling(labeledPoints, rawModel)

        coordinate.trainModel(initialModel)

      case None =>
        doReturn(updatedModel).when(optimizationProblem).runWithSampling(labeledPoints)

        coordinate.trainModel()
    }
    val newFixedEffectModel = newModel.asInstanceOf[FixedEffectModel]

    assertTrue(newFixedEffectModel.modelBroadcast.eq(modelBroadcast))
  }

  /**
   * Test that a [[FixedEffectCoordinate]] can score data using a (mocked) [[FixedEffectModel]].
   */
  @Test
  def testScore(): Unit = {

    val dataset = mock(classOf[FixedEffectDataset])
    val labeledPoints = mock(classOf[RDD[(UniqueSampleId, LabeledPoint)]])
    val scores = mock(classOf[RDD[(UniqueSampleId, Double)]])
    val fixedEffectModel = mock(classOf[FixedEffectModel])
    val modelBroadcast = mock(classOf[Broadcast[GeneralizedLinearModel]])
    val optimizationProblem = mock(classOf[DistributedOptimizationProblem[DistributedObjectiveFunction]])

    doReturn(labeledPoints).when(dataset).labeledPoints
    doReturn(scores).when(labeledPoints).mapValues(Matchers.any())

    doReturn(modelBroadcast).when(fixedEffectModel).modelBroadcast

    val coordinate = new FixedEffectCoordinate(dataset, optimizationProblem)
    val coordinateDataScores = coordinate.score(fixedEffectModel)

    assertTrue(coordinateDataScores.scoresRdd.eq(scores))
  }
}

object FixedEffectCoordinateTest {

  class MockFixedEffectCoordinate[Objective <: DistributedObjectiveFunction](
      val publicDataset: FixedEffectDataset,
      val publicOptimizationProblem: DistributedOptimizationProblem[Objective])
    extends FixedEffectCoordinate[Objective](publicDataset, publicOptimizationProblem) {

    override protected[algorithm] def updateCoordinateWithDataset(
        newDataset: FixedEffectDataset): MockFixedEffectCoordinate[Objective] =
      new MockFixedEffectCoordinate(newDataset, publicOptimizationProblem)
  }
}
