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

import org.mockito.Mockito._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.Dataset
import com.linkedin.photon.ml.model.DatumScoringModel

/**
 * Unit tests for [[ModelCoordinate]].
 */
class ModelCoordinateTest {

  import ModelCoordinateTest._

  /**
   * Test that attempts to update a [[ModelCoordinate]] will fail.
   */
  @Test(expectedExceptions = Array(classOf[UnsupportedOperationException]))
  def testUpdateCoordinateWithDataset(): Unit = {

    val mockModelCoordinate = mock(classOf[ModelCoordinate[MockDataset]])
    val mockDataset = mock(classOf[MockDataset])

    doCallRealMethod().when(mockModelCoordinate).updateCoordinateWithDataset(mockDataset)

    mockModelCoordinate.updateCoordinateWithDataset(mockDataset)
  }

  @DataProvider
  def trainModelInput: Array[Array[Any]] = {

    val mockModelCoordinate = mock(classOf[ModelCoordinate[MockDataset]])
    val mockInitialModel = mock(classOf[DatumScoringModel])

    doCallRealMethod().when(mockModelCoordinate).trainModel(mockInitialModel)
    doCallRealMethod().when(mockModelCoordinate).trainModel()

    Array(
      Array(mockModelCoordinate, Some(mockInitialModel)),
      Array(mockModelCoordinate, None))
  }

  /**
   * Test that attempts to train a new model for a [[ModelCoordinate]] will fail.
   *
   * @param modelCoordinate A [[ModelCoordinate]]
   * @param initialModelOpt An optional existing model to use as a starting point for optimization
   */
  @Test(dataProvider = "trainModelInput", expectedExceptions = Array(classOf[UnsupportedOperationException]))
  def testTrainModel(
      modelCoordinate: ModelCoordinate[MockDataset],
      initialModelOpt: Option[DatumScoringModel]): Unit = initialModelOpt match {

    case Some(initialModel) =>
      modelCoordinate.trainModel(initialModel)

    case None =>
      modelCoordinate.trainModel()
  }
}

object ModelCoordinateTest {

  abstract class MockDataset extends Dataset[MockDataset]
}
