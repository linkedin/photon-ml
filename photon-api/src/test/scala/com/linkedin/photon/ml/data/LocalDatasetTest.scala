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

import java.util.Random

import breeze.linalg.Vector
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst

/**
 * Unit tests for [[LocalDataset]].
 */
class LocalDatasetTest {

  /**
   * Test that the factory function for [[LocalDataset]] works as intended.
   */
  @Test
  def testApply(): Unit = {

    val mockLabeledPoint = mock(classOf[LabeledPoint])
    val mockFeatures = mock(classOf[Vector[Double]])

    doReturn(mockFeatures).when(mockLabeledPoint).features
    doReturn(1).when(mockFeatures).length

    val dataArray = Array[Long](5L, 4L, 3L, 2L, 1L).map((_, mockLabeledPoint))
    val localDatasetUnsorted = LocalDataset(dataArray, isSortedByFirstIndex = true)
    val localDatasetSorted = LocalDataset(dataArray, isSortedByFirstIndex = false)

    localDatasetUnsorted
      .dataPoints
      .foldLeft(Integer.MAX_VALUE.toLong) { case (prevId, (id, _)) =>
        assertTrue(prevId > id)

        id
      }
    localDatasetSorted
      .dataPoints
      .foldLeft(Integer.MIN_VALUE.toLong) { case (prevId, (id, _)) =>
        assertTrue(prevId < id)

        id
      }
  }

  /**
   * Test that offset values of each sample in a [[LocalDataset]] can be correctly modified.
   */
  @Test(dependsOnMethods = Array("testApply"))
  def testAddScoresToOffsets(): Unit = {

    val mockVector = mock(classOf[Vector[Double]])

    val random = new Random(MathConst.RANDOM_SEED)
    val labeledPoints = (1L to 5L).toArray.map((_, LabeledPoint(1D, mockVector)))
    val offsets = labeledPoints.map { case (uid, _) =>
      (uid, random.nextDouble())
    }
    val localDataset = LocalDataset(labeledPoints)
    val updatedLocalDataset = localDataset.addScoresToOffsets(offsets)

    localDataset
      .dataPoints
      .foreach { case (_, labeledPoint) =>
        assertEquals(labeledPoint.offset, 0D, MathConst.EPSILON)
      }
    updatedLocalDataset
      .dataPoints
      .map(_._2)
      .zip(offsets.map(_._2))
      .foreach { case (labeledPoint, offset) =>
        assertEquals(labeledPoint.offset, offset, MathConst.EPSILON)
      }
  }
}
