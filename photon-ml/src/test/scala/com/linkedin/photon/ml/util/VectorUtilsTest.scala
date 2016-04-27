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
package com.linkedin.photon.ml.util

import breeze.linalg.{DenseVector, SparseVector}
import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Simple tests for functions in [[VectorUtils]]
 * @todo Test [[VectorUtils.kroneckerProduct()]]
 */
class VectorUtilsTest {
  @Test
  def testConvertIndexAndValuePairArrayToSparseVector() = {
    val length = 6

    // With empty data
    val emptyIndexAndData = Array[(Int, Double)]()
    val emptySparseVector = VectorUtils.convertIndexAndValuePairArrayToSparseVector(emptyIndexAndData, length)
    assertEquals(emptySparseVector.length, length)
    assertEquals(emptySparseVector.activeSize, emptyIndexAndData.length)

    // Normal case
    val normalIndexAndData = Array[(Int, Double)](0 -> 0.0, 3 -> 3.0, 5 -> 5.0)
    val normalSparseVector = VectorUtils.convertIndexAndValuePairArrayToSparseVector(normalIndexAndData, length)
    assertEquals(normalSparseVector.length, length)
    assertEquals(normalSparseVector.activeSize, normalIndexAndData.length)
    normalIndexAndData.foreach { case (index, data) =>
      assertEquals(data, normalSparseVector(index))
    }
  }

  @Test
  def testConvertIndexAndValuePairArrayToDenseVector() = {
    val length = 6

    // With empty data
    val emptyIndexAndData = Array[(Int, Double)]()
    val emptyDenseVector = VectorUtils.convertIndexAndValuePairArrayToDenseVector(emptyIndexAndData, length)
    assertEquals(emptyDenseVector.length, length)
    assertEquals(emptyDenseVector.activeSize, length)

    // Normal case
    val normalIndexAndData = Array[(Int, Double)](0 -> 0.0, 3 -> 3.0, 5 -> 5.0)
    val normalDenseVector = VectorUtils.convertIndexAndValuePairArrayToDenseVector(normalIndexAndData, length)
    assertEquals(normalDenseVector.length, length)
    assertEquals(normalDenseVector.activeSize, length)
    normalIndexAndData.foreach { case (index, data) =>
      assertEquals(data, normalDenseVector(index))
    }
  }

  @Test
  def testConvertIndexAndValuePairArrayToVector() = {
    val length = 6

    // For sparse vector
    val activeSizeForSparseVector = math.floor(length * VectorUtils.SPARSE_VECTOR_ACTIVE_SIZE_TO_SIZE_RATIO - 1).toInt
    val indexAndDataForSparseVector = Array.tabulate[(Int, Double)](activeSizeForSparseVector)(i => (i, 1.0))
    val sparseVector = VectorUtils.convertIndexAndValuePairArrayToVector(indexAndDataForSparseVector, length)
    assertTrue(sparseVector.isInstanceOf[SparseVector[Double]])
    assertEquals(sparseVector.activeSize, activeSizeForSparseVector)
    assertEquals(sparseVector.length, length)

    // For dense vector
    val activeSizeForDenseVector = math.floor(length * VectorUtils.SPARSE_VECTOR_ACTIVE_SIZE_TO_SIZE_RATIO + 1).toInt
    val indexAndDataForDenseVector = Array.tabulate[(Int, Double)](activeSizeForDenseVector)(i => (i, 1.0))
    val denseVector = VectorUtils.convertIndexAndValuePairArrayToVector(indexAndDataForDenseVector, length)
    assertTrue(denseVector.isInstanceOf[DenseVector[Double]])
    assertEquals(denseVector.activeSize, length)
    assertEquals(denseVector.length, length)
  }
}
