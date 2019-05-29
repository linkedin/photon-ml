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
package com.linkedin.photon.ml.projector

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.util.VectorUtils

/**
 *
 */
class LinearSubspaceProjectorTest {

  import LinearSubspaceProjectorTest._

  /**
   *
   */
  @Test
  def testBuilder(): Unit = {

    assertEquals(PROJECTOR.originalToProjectedSpaceMap.size, PROJECTED_DIMENSION)
    assertEquals(PROJECTOR.originalToProjectedSpaceMap.keySet, Set[Int](0, 1, 4, 5, 6, 7, 9))
    assertEquals(PROJECTOR.originalSpaceDimension, ORIGINAL_DIMENSION)
    assertEquals(PROJECTOR.projectedSpaceDimension, PROJECTED_DIMENSION)
  }

  /**
   *
   */
  @Test
  def testProjectForward(): Unit = {

    val fV = new SparseVector[Double](Array(0, 2, 4, 5, 8), Array(1.0, 2.0, 4.0, 5.0, 8.0), ORIGINAL_DIMENSION)
    val projectedFV = PROJECTOR.projectForward(fV)
    val expectedActiveTuples = Set[(Int, Double)](
      (PROJECTOR.originalToProjectedSpaceMap(0), 1.0),
      (PROJECTOR.originalToProjectedSpaceMap(4), 4.0),
      (PROJECTOR.originalToProjectedSpaceMap(5), 5.0))

    // Check that the new vector is in the projected space
    assertEquals(projectedFV.length, PROJECTED_DIMENSION)

    // Check that the features present in the vector and in the mapping have been projected, and that the features
    // present but not in the mapping have not
    assertEquals(getActiveTuples(projectedFV), expectedActiveTuples)
  }

  /**
   *
   */
  @Test
  def testProjectBackward(): Unit = {
    val fV = new DenseVector[Double](Array(0.0, 0.1, 0.2, 0.3, 0.0, 0.5, 0.6))
    val expectedActiveTuples = Set[(Int, Double)](
      (PROJECTOR.projectedToOriginalSpaceMap(1), 0.1),
      (PROJECTOR.projectedToOriginalSpaceMap(2), 0.2),
      (PROJECTOR.projectedToOriginalSpaceMap(3), 0.3),
      (PROJECTOR.projectedToOriginalSpaceMap(5), 0.5),
      (PROJECTOR.projectedToOriginalSpaceMap(6), 0.6))
    val projectedFV = PROJECTOR.projectBackward(fV)

    // Check that the new vector is in the original space
    assertEquals(projectedFV.length, ORIGINAL_DIMENSION)

    // Check that the features present in the vector have been projected, and that the features missing from the vector
    // and those that are not active in the mapping are 0
    assertEquals(getActiveTuples(projectedFV), expectedActiveTuples)
  }
}

object LinearSubspaceProjectorTest {

  private val ORIGINAL_DIMENSION = 10
  private val PROJECTED_DIMENSION = 7
  private val PROJECTOR = buildLinearSubspaceProjector(
    List[Vector[Double]](
      new SparseVector[Double](Array(0, 4, 6, 7, 9), Array(1.0, 4.0, 6.0, 7.0, 9.0), 10),
      new SparseVector[Double](Array(0, 1), Array(1.0, 1.0), 10),
      new DenseVector[Double](Array(0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 0.0, 0.0))))

  /**
   * Generate a [[LinearSubspaceProjector]] given an iterator of feature vectors.
   *
   * @param features An [[Iterable]] of feature vectors
   * @return The generated projection map
   */
  private def buildLinearSubspaceProjector(features: Iterable[Vector[Double]]): LinearSubspaceProjector = {

    val originalSpaceDimension = features.head.length
    val originalToProjectedSpaceMap = features
      .flatMap(VectorUtils.getActiveIndices)
      .toSet

    new LinearSubspaceProjector(originalToProjectedSpaceMap, originalSpaceDimension)
  }

  /**
   *
   * @param vector
   * @return
   */
  def getActiveTuples(vector: Vector[Double]): Set[(Int, Double)] =
    vector.iterator.filter(x => math.abs(x._2) > 0.0).toSet
}
