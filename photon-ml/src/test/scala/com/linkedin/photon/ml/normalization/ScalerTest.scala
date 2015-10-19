package com.linkedin.photon.ml.normalization

import java.lang.{Object => JObject}

import breeze.linalg.{DenseVector, SparseVector, Vector}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.test.{Assertions, SparkTestUtils}
import org.testng.Assert._
import Assertions.assertIterableEqualsWithTolerance
import org.testng.annotations.{DataProvider, Test}


/**
 * Test [[VectorScaler]] and [[LabeledPointTransformer]]
 * @author dpeng
 */
class ScalerTest extends SparkTestUtils {
  private val _delta = 1.0E-9
  private val _factors = DenseVector[Double](1.0, 0.5, 4.5, -1.0, -0.4, -5.4, 0)
  private val _size = _factors.size
  private val _vectorScaler = new VectorScaler(_factors)
  private val _labeldPointScaler = new LabeledPointTransformer(_vectorScaler)

  private val _denseTestVector = DenseVector[Double](1.2, 0.5, -2.0, 4.1, -1.2, 0, 7)
  private val _denseTransformedVector = DenseVector[Double](1.2, 0.25, -9.0, -4.1, 0.48, 0, 0)
  private val _sparseTestVector = new SparseVector[Double](Array(1, 4), Array(2.1, 0.5), _size)
  private val _sparseTransformedVector = new SparseVector[Double](Array(1, 4), Array(1.05, -0.2), _size)

  private val _denseTestLabeledPoint = new LabeledPoint(0, _denseTestVector)
  private val _denseTransformedLabeledPoint = new LabeledPoint(0, _denseTransformedVector)
  private val _sparseTestLabeledPoint = new LabeledPoint(0, _sparseTestVector)
  private val _sparseTransformedLabeledPoint = new LabeledPoint(0, _sparseTransformedVector)

  private def assertLabeledPointEqual(actual: LabeledPoint, expected: LabeledPoint): Unit = {
    assertEquals(actual.label, expected.label, _delta)
    assertEquals(actual.offset, expected.offset, _delta)
    assertEquals(actual.weight, expected.weight, _delta)
    assertIterableEqualsWithTolerance(actual.features.toArray, expected.features.toArray, _delta)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testDenseVectorWithWrongDimension(): Unit = {
    val testVector = DenseVector.ones[Double](4)
    _vectorScaler.transform(testVector)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testSparseVectorWithWrongDimension(): Unit = {
    val testVector = SparseVector.zeros[Double](4)
    _vectorScaler.transform(testVector)
  }

  @Test
  def testDenseVectorOnes(): Unit = {
    val testVector = DenseVector.ones[Double](_size)
    val transformedVector = _vectorScaler.transform(testVector)
    assertEquals(transformedVector, _factors)
  }

  @Test
  def testSparseVectorZeros(): Unit = {
    val testVector = SparseVector.zeros[Double](_size)
    val transformedVector = _vectorScaler.transform(testVector)
    assertEquals(transformedVector, testVector)
  }

  @Test(dataProvider = "vectorsProvider")
  def testGeneralVectors(input: Vector[Double], output: Vector[Double]): Unit = {
    val copy = input.copy
    val transformedVector = _vectorScaler.transform(input)
    assertEquals(transformedVector, output)
    // Sanity check: the transformation doesn't modify the original vector
    assertEquals(input, copy)
    // Sanity check: the transformed vector and the original vector are of the same class
    assertEquals(transformedVector.getClass, input.getClass)
  }

  @DataProvider(parallel = true)
  def vectorsProvider(): Array[Array[JObject]] = {
    Array(
      Array(_denseTestVector, _denseTransformedVector),
      Array(_sparseTestVector, _sparseTransformedVector)
    )
  }

  @DataProvider(parallel = true)
  def labeledPointsProvider(): Array[Array[JObject]] = {
    Array(
      Array(_denseTestLabeledPoint, _denseTransformedLabeledPoint),
      Array(_sparseTestLabeledPoint, _sparseTransformedLabeledPoint)
    )
  }

  @Test(dataProvider = "labeledPointsProvider")
  def testLabeledPointsProvider(input: LabeledPoint, output: LabeledPoint): Unit = {
    val actual = _labeldPointScaler.transform(input)
    assertLabeledPointEqual(actual, output)
  }

}
