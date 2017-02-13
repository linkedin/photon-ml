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
package com.linkedin.photon.ml.projector

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.testng.Assert
import org.testng.annotations.Test

class IndexMapProjectorTest {

  private val projector = IndexMapProjector.buildIndexMapProjector(
    List[Vector[Double]](
      new SparseVector[Double](Array(0, 4, 6, 7, 9), Array(1.0, 4.0, 6.0, 7.0, 9.0), 10),
      new SparseVector[Double](Array(0, 1), Array(1.0, 1.0), 10),
      new SparseVector[Double](Array(4, 5, 7), Array(4.0, 5.0, 7.0), 10)))

  @Test
  def testBuilder(): Unit = {
    Assert.assertEquals(projector.originalToProjectedSpaceMap.size, 7)
    Assert.assertEquals(projector.originalToProjectedSpaceMap.keySet, Set[Int](0, 1, 4, 5, 6, 7, 9))
    Assert.assertEquals(projector.originalSpaceDimension, 10)
    Assert.assertEquals(projector.projectedSpaceDimension, 7)
  }

  @Test
  def testProjectFeatures(): Unit = {
    val fv = new SparseVector[Double](Array(0, 2, 4, 5, 8), Array(1.0, 2.0, 4.0, 5.0, 8.0), 10)
    val expectedActiveTuples = Set[(Int, Double)]((projector.originalToProjectedSpaceMap(0), 1.0),
      (projector.originalToProjectedSpaceMap(4), 4.0),
      (projector.originalToProjectedSpaceMap(5), 5.0))
    val projected = projector.projectFeatures(fv)

    // ensure it is in the expected space
    Assert.assertEquals(projected.length, 7)

    // ensure active tuples are the ones expected and non-existent features are ignored
    Assert.assertEquals(projected.iterator
      .filter(x => math.abs(x._2) > 0.0)
      .toSet, expectedActiveTuples)
  }

  @Test
  def testProjectCoefficients(): Unit = {
    val coefficients = new DenseVector[Double](Array(0.0, 0.1, 0.2, 0.3, 0.0, 0.5, 0.6))
    val expectedActiveTuples = Set[(Int, Double)]((projector.projectedToOriginalSpaceMap(1), 0.1),
      (projector.projectedToOriginalSpaceMap(2), 0.2),
      (projector.projectedToOriginalSpaceMap(3), 0.3),
      (projector.projectedToOriginalSpaceMap(5), 0.5),
      (projector.projectedToOriginalSpaceMap(6), 0.6))
    val projected = projector.projectCoefficients(coefficients)

    // ensure it is back in the original space
    Assert.assertEquals(projected.length, 10)
    // ensure active tuples are the ones expected
    Assert.assertEquals(projected.iterator
      .filter(x => math.abs(x._2) > 0.0)
      .toSet, expectedActiveTuples)
  }
}