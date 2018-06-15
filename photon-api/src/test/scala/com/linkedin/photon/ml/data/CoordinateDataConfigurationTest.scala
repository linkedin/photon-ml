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
package com.linkedin.photon.ml.data

import org.mockito.Mockito._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.projector.ProjectorType

/**
 * Tests for [[CoordinateDataConfiguration]].
 */
class CoordinateDataConfigurationTest {

  @DataProvider
  def invalidInput(): Array[Array[Any]] = Array(
    Array(-1, 1, 1, 1, 1D),
    Array(1, -1, 1, 1, 1D),
    Array(1, 1, -1, 1, 1D),
    Array(1, 1, 1, -1, 1D),
    Array(1, 1, 1, 1, -1D),
    Array(1, 2, 1, 1, 1D))

  /**
   * Test that invalid input will be rejected.
   *
   * @param minPartitions The minimum number of data partitions
   * @param activeLowerBound The lower bound on number of active data samples
   * @param activeUpperBound The upper bound on number of active data samples
   * @param passiveLowerBound The lower bound on number of passive data samples
   * @param featuresToSamplesRatio The upper bound on the ratio between data samples and features
   */
  @Test(dataProvider = "invalidInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testSetupWithInvalidInput(
      minPartitions: Int,
      activeLowerBound: Int,
      activeUpperBound: Int,
      passiveLowerBound: Int,
      featuresToSamplesRatio: Double): Unit = {

    val mockREType = "reType"
    val mockFeatureShardId = "featureShardId"
    val mockProjector = mock(classOf[ProjectorType])

    RandomEffectDataConfiguration(
      mockREType,
      mockFeatureShardId,
      minPartitions,
      Some(activeLowerBound),
      Some(activeUpperBound),
      Some(passiveLowerBound),
      Some(featuresToSamplesRatio),
      mockProjector)
  }

  @DataProvider
  def validInput(): Array[Array[Any]] = Array(
    Array(1, 1, 1, 1, 1D),
    Array(1, 2, 3, 4, 5D))

  /**
   * Test that valid input will not be rejected.
   *
   * @param minPartitions The minimum number of data partitions
   * @param activeLowerBound The lower bound on number of active data samples
   * @param activeUpperBound The upper bound on number of active data samples
   * @param passiveLowerBound The lower bound on number of passive data samples
   * @param featuresToSamplesRatio The upper bound on the ratio between data samples and features
   */
  @Test(dataProvider = "validInput")
  def testSetupWithValidInput(
      minPartitions: Int,
      activeLowerBound: Int,
      activeUpperBound: Int,
      passiveLowerBound: Int,
      featuresToSamplesRatio: Double): Unit = {

    val mockREType = "reType"
    val mockFeatureShardId = "featureShardId"
    val mockProjector = mock(classOf[ProjectorType])

    RandomEffectDataConfiguration(
      mockREType,
      mockFeatureShardId,
      minPartitions,
      Some(activeLowerBound),
      Some(activeUpperBound),
      Some(passiveLowerBound),
      Some(featuresToSamplesRatio),
      mockProjector)
  }
}
