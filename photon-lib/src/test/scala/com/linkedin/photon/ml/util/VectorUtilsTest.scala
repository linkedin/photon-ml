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
package com.linkedin.photon.ml.util

import scala.util.Random

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SparkVector}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

// TODO: Test [[VectorUtils.kroneckerProduct()]]
/**
 * Simple tests for functions in [[VectorUtils]].
 */
class VectorUtilsTest {

  private val seed = 7
  private val random = new Random(seed)

  @Test
  def testConvertIndexAndValuePairArrayToSparseVector(): Unit = {
    val length = 6

    // With empty data
    val emptyIndexAndData = Array[(Int, Double)]()
    val emptySparseVector = VectorUtils.toSparseVector(emptyIndexAndData, length)
    assertEquals(emptySparseVector.length, length)
    assertEquals(emptySparseVector.activeSize, emptyIndexAndData.length)

    // Normal case
    val normalIndexAndData = Array[(Int, Double)](0 -> 0.0, 3 -> 3.0, 5 -> 5.0)
    val normalSparseVector = VectorUtils.toSparseVector(normalIndexAndData, length)
    assertEquals(normalSparseVector.length, length)
    assertEquals(normalSparseVector.activeSize, normalIndexAndData.length)
    normalIndexAndData.foreach { case (index, data) =>
      assertEquals(data, normalSparseVector(index))
    }
  }

  @Test
  def testConvertIndexAndValuePairArrayToDenseVector(): Unit = {
    val length = 6

    // With empty data
    val emptyIndexAndData = Array[(Int, Double)]()
    val emptyDenseVector = VectorUtils.toDenseVector(emptyIndexAndData, length)
    assertEquals(emptyDenseVector.length, length)
    assertEquals(emptyDenseVector.activeSize, length)

    // Normal case
    val normalIndexAndData = Array[(Int, Double)](0 -> 0.0, 3 -> 3.0, 5 -> 5.0)
    val normalDenseVector = VectorUtils.toDenseVector(normalIndexAndData, length)
    assertEquals(normalDenseVector.length, length)
    assertEquals(normalDenseVector.activeSize, length)
    normalIndexAndData.foreach { case (index, data) =>
      assertEquals(data, normalDenseVector(index))
    }
  }

  @Test
  def testConvertIndexAndValuePairArrayToVector(): Unit = {
    val length = 6

    // For sparse vector
    val activeSizeForSparseVector = math.floor(length * VectorUtils.SPARSE_VECTOR_ACTIVE_SIZE_TO_SIZE_RATIO - 1).toInt
    val indexAndDataForSparseVector = Array.tabulate[(Int, Double)](activeSizeForSparseVector)(i => (i, 1.0))
    val sparseVector = VectorUtils.toVector(indexAndDataForSparseVector, length)
    assertTrue(sparseVector.isInstanceOf[SparseVector[Double]])
    assertEquals(sparseVector.activeSize, activeSizeForSparseVector)
    assertEquals(sparseVector.length, length)

    // For dense vector
    val activeSizeForDenseVector = math.floor(length * VectorUtils.SPARSE_VECTOR_ACTIVE_SIZE_TO_SIZE_RATIO + 1).toInt
    val indexAndDataForDenseVector = Array.tabulate[(Int, Double)](activeSizeForDenseVector)(i => (i, 1.0))
    val denseVector = VectorUtils.toVector(indexAndDataForDenseVector, length)
    assertTrue(denseVector.isInstanceOf[DenseVector[Double]])
    assertEquals(denseVector.activeSize, length)
    assertEquals(denseVector.length, length)
  }

  @DataProvider
  def featureValsProvider(): Array[Array[Array[Double]]] = {
    val n = 10
    val dim = 100
    val mean = 0.0
    val sd = 1.0

    (0 until n).map { i =>
      Array(Seq.fill(dim)({ (random.nextGaussian * sd) + mean }).toArray)
    }.toArray
  }

  @Test(dataProvider = "featureValsProvider")
  def testSparseBreezeToMllib(featureVals: Array[Double]): Unit = {
    val dim = featureVals.length
    val indexes = (0 until dim).toArray
    val vector = new SparseVector(indexes, featureVals, dim)
    val converted = VectorUtils.breezeToMllib(vector)

    converted match {
      case v: SSV =>
        assertVectorsEqual(vector, converted)

      case v => fail(s"Unexpected vector type: ${v.getClass.getName}")
    }
  }

  @Test(dataProvider = "featureValsProvider")
  def testDenseBreezeToMllib(featureVals: Array[Double]): Unit = {
    val vector = new DenseVector(featureVals)
    val converted = VectorUtils.breezeToMllib(vector)

    converted match {
      case v: SDV =>
        assertVectorsEqual(vector, converted)

      case v => fail(s"Unexpected vector type: ${v.getClass.getName}")
    }
  }

  @Test(dataProvider = "featureValsProvider")
  def testSparseMllibToBreeze(featureVals: Array[Double]): Unit = {
    val dim = featureVals.length
    val indexes = (0 until dim).toArray
    val vector = new SSV(dim, indexes, featureVals)
    val converted = VectorUtils.mllibToBreeze(vector)

    converted match {
      case v: SparseVector[Double] =>
        assertVectorsEqual(converted, vector)

      case v => fail(s"Unexpected vector type: ${v.getClass.getName}")
    }
  }

  @Test(dataProvider = "featureValsProvider")
  def testDenseMllibToBreeze(featureVals: Array[Double]): Unit = {
    val vector = new SDV(featureVals)
    val converted = VectorUtils.mllibToBreeze(vector)

    converted match {
      case v: DenseVector[Double] =>
        assertVectorsEqual(converted, vector)

      case v => fail(s"Unexpected vector type: ${v.getClass.getName}")
    }
  }

  /**
    * Asserts that the breeze and spark vectors are equal.
    *
    * @param breezeVector A Breeze vector
    * @param sparkVector A Spark vector
    */
  def assertVectorsEqual(breezeVector: Vector[Double], sparkVector: SparkVector): Unit = {
    assertEquals(breezeVector.size, sparkVector.size)
    for (i <- 0 until breezeVector.size) {
      assertEquals(breezeVector(i), sparkVector(i))
    }
  }
}
