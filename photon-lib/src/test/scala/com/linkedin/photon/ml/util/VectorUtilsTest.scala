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
import org.apache.spark.mllib.linalg.{DenseVector => SparkDenseVector, SparseVector => SparkSparseVector, Vector => SparkVector}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

// TODO: Test [[VectorUtils.kroneckerProduct()]]

/**
 * Simple tests for functions in [[VectorUtils]].
 */
class VectorUtilsTest {

  import VectorUtilsTest._

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
    val dim = featureVals.length + 1
    val (vals, indexes) = featureVals
      .zipWithIndex
      .flatMap( pair => if (random.nextBoolean()) Seq(pair) else Seq())
      .unzip
    val vector = new SparseVector[Double](indexes.toArray, vals.toArray, dim)
    val converted = VectorUtils.breezeToMllib(vector)

    converted match {
      case v: SparkSparseVector =>
        assertVectorsEqual(vector, converted)

      case v => fail(s"Unexpected vector type: ${v.getClass.getName}")
    }
  }

  @Test(dataProvider = "featureValsProvider")
  def testDenseBreezeToMllib(featureVals: Array[Double]): Unit = {
    val vector = new DenseVector(featureVals)
    val converted = VectorUtils.breezeToMllib(vector)

    converted match {
      case v: SparkDenseVector =>
        assertVectorsEqual(vector, converted)

      case v => fail(s"Unexpected vector type: ${v.getClass.getName}")
    }
  }

  @Test(dataProvider = "featureValsProvider")
  def testSparseMllibToBreeze(featureVals: Array[Double]): Unit = {
    val dim = featureVals.length + 1
    val (vals, indexes) = featureVals
      .zipWithIndex
      .flatMap( pair => if (random.nextBoolean()) Seq(pair) else Seq())
      .unzip
    val vector = new SparkSparseVector(dim, indexes.toArray, vals.toArray)
    val converted = VectorUtils.mllibToBreeze(vector)

    converted match {
      case v: SparseVector[Double] =>
        assertVectorsEqual(converted, vector)

      case v => fail(s"Unexpected vector type: ${v.getClass.getName}")
    }
  }

  @Test(dataProvider = "featureValsProvider")
  def testDenseMllibToBreeze(featureVals: Array[Double]): Unit = {
    val vector = new SparkDenseVector(featureVals)
    val converted = VectorUtils.mllibToBreeze(vector)

    converted match {
      case v: DenseVector[Double] =>
        assertVectorsEqual(converted, vector)

      case v => fail(s"Unexpected vector type: ${v.getClass.getName}")
    }
  }

  @DataProvider
  def activeIndicesProvider() = Array(
    Array(DenseVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Set(0, 1, 2, 3, 4, 5)),
    Array(DenseVector(0.0, 1.0, 0.0, 2.0, 3.0, 0.0), Set(1, 3, 4)),
    Array(DenseVector(1e-18, 1.0, 0.0, 2.0, 3.0, 0.0), Set(1, 3, 4)),
    Array(new SparseVector(Array(1, 3, 4), Array(1.0, 2.0, 3.0), 3), Set(1, 3, 4))
  )

  @Test(dataProvider = "activeIndicesProvider")
  def testGetActiveIndices(vector: Vector[Double], expected: Set[Int]): Unit = {
    val result = VectorUtils.getActiveIndices(vector)
    assertEquals(result, expected)
  }

  /**
   * Asserts that the breeze and spark vectors are equal.
   *
   * @param breezeVector A Breeze vector
   * @param sparkVector A Spark vector
   */
  def assertVectorsEqual(breezeVector: Vector[Double], sparkVector: SparkVector): Unit = {

    assertEquals(breezeVector.length, sparkVector.size)
    for (i <- 0 until breezeVector.length) {
      assertEquals(breezeVector(i), sparkVector(i))
    }
  }

  /**
   * Test generation of Breeze zero vectors.
   */
  @Test
  def testInitializeZerosVectorOfSameType(): Unit = {

    val r: Random = new Random(RANDOM_SEED)

    //
    // Dense prototype vector
    //

    val prototypeDenseVector = DenseVector.fill(VECTOR_DIMENSION)(r.nextDouble())
    val initializedDenseVector = VectorUtils.zeroOfSameType(prototypeDenseVector)

    assertEquals(prototypeDenseVector.length, initializedDenseVector.length,
      s"Length of the initialized vector (${initializedDenseVector.length}) " +
        s"is different from the prototype vector (${initializedDenseVector.length}})")
    assertTrue(initializedDenseVector.isInstanceOf[DenseVector[Double]],
      s"The initialized dense vector (${initializedDenseVector.getClass}), " +
        s"is not an instance of the prototype vectors' class (${prototypeDenseVector.getClass})")

    //
    // Sparse prototype vector
    //

    val indices = Array.tabulate[Int](VECTOR_DIMENSION)(i => i).filter(_ => r.nextBoolean())
    val values = indices.map(_ => r.nextDouble())
    val prototypeSparseVector = new SparseVector[Double](indices, values, VECTOR_DIMENSION)
    val initializedSparseVector = VectorUtils.zeroOfSameType(prototypeSparseVector)

    assertEquals(prototypeSparseVector.length, initializedSparseVector.length,
      s"Length of the initialized vector (${initializedSparseVector.length}) " +
        s"is different from the prototype vector (${prototypeSparseVector.length}})")
    assertTrue(initializedSparseVector.isInstanceOf[SparseVector[Double]],
      s"The initialized sparse vector (${initializedSparseVector.getClass}) " +
        s"is not an instance of the prototype vectors' class (${prototypeSparseVector.getClass})")
  }

  /**
   * Test that zero vector generation will fail for unsupported vector types.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInitializeZerosVectorOfSameTypeOfUnsupportedVectorType(): Unit =
    VectorUtils.zeroOfSameType(new MockVector[Double]())

  @DataProvider
  def invertVectorWithZeroHandlerProvider() = Array(
    Array(DenseVector(1.0, 0.0, 2.0), 3.0, DenseVector(1.0, 3.0, 0.5)),
    Array(new SparseVector(Array(1, 2), Array(1.0, 2.0), 3), 4.0, DenseVector(4.0, 1.0, 0.5))
  )

  /**
   * Test inverting vectors with zero handler.
   */
  @Test(dataProvider = "invertVectorWithZeroHandlerProvider")
  def testInvertVectorWithZeroHandler(vector: Vector[Double], replacedVal: Double, expectedVector: Vector[Double]): Unit =
    assertEquals(VectorUtils.invertVectorWithZeroHandler(vector, replacedVal), expectedVector)
}

object VectorUtilsTest {

  private val VECTOR_DIMENSION: Int = 10
  private val RANDOM_SEED: Long = 1234567890L

  /**
   * This is a Vector that mocks a different implementation of breeze Vector, it does nothing meaningful.
   *
   * @tparam V Some data type (irrelevant for this mock)
   */
  private class MockVector[V] extends Vector[V] {
    override def length: Int = 0

    override def copy: Vector[V] = null

    override def update(i: Int, v: V): Unit = {}

    override def activeSize: Int = 0

    override def apply(i: Int): V = 0d.asInstanceOf[V]

    override def activeIterator: Iterator[(Int, V)] = null

    override def activeKeysIterator: Iterator[Int] = null

    override def activeValuesIterator: Iterator[V] = null

    override def repr: Vector[V] = null
  }
}
