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

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}

/**
 * Test the functions in [[LabeledPoint]].
 */
class LabeledPointTest extends SparkTestUtils {
  import LabeledPointTest._

  /**
   *
   * @return
   */
  def generateBenignLocalDataSetBinaryClassification(): List[LabeledPoint] = {
    drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      DATA_RANDOM_SEED,
      TRAINING_SAMPLES,
      PROBLEM_DIMENSION)
      .map { obj =>
        assertEquals(obj._2.length, PROBLEM_DIMENSION, "Samples should have expected lengths")
        LabeledPoint(label = obj._1, features = obj._2, offset = 1.5, weight = 1.0)
      }
      .toList
  }

  @DataProvider(name = "getDataPoints")
  def getDataPoints: Array[Array[Any]] = {
    Array(
      Array(DenseVector(1.0, 0.0, 0.4, 0.5), 12.34, DenseVector(0.0, 0.0, 0.0, 0.0), 12.34),
      Array(DenseVector(1.0, 0.0, 0.4, 0.5), 1.0, DenseVector(-1.0, -0.5, 0.1, 0.0), 0.04),
      Array(SparseVector(4)((0, 1.0), (2, 0.4), (3, 0.5)), -1.0, DenseVector(-1.0, -0.5, 0.1, 0.0), -1.96),
      Array(DenseVector(1.0, 0.0, 0.4, 0.5), 0.0, SparseVector(4)((0, -1.0), (1, -0.5), (2, 0.1)), -0.96),
      Array(SparseVector(4)((0, 1.0), (2, 0.4), (3, 0.5)), -100.0, SparseVector(4)((0, -1.0), (1, -0.5), (2, 0.1)), -100.96)
    )
  }

  @DataProvider
  def getRandomDataPoints: Array[Array[LabeledPoint]] = {
    val randomSamples = generateBenignLocalDataSetBinaryClassification()
    randomSamples.map(Array(_)).toArray
  }

  /**
   * Test the class and object.
   */
  @Test
  def testApply(): Unit = {
    val label = 1.0
    val features = DenseVector[Double](1.0, 10.0, 0.0, -100.0)
    val offset = 1.5
    val weight = 1.0
    val dataPoint = new LabeledPoint(label, features, offset, weight)
    val expected = LabeledPoint(label, features, offset, weight)
    assertEquals(dataPoint.label, expected.label, TOLERANCE)
    assertIterableEqualsWithTolerance(dataPoint.features.toArray, expected.features.toArray, TOLERANCE)
    assertEquals(dataPoint.offset, expected.offset, TOLERANCE)
    assertEquals(dataPoint.weight, expected.weight, TOLERANCE)
  }

  /**
   * Test unapply.
   */
  @Test
  def testUnapply(): Unit = {
    val label = 1.0
    val features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
    val offset = 1.5
    val weight = 3.2
    val dataPoint = LabeledPoint(label, features, offset, weight)
    val params = LabeledPoint.unapply(dataPoint)
    assertEquals(params.get._1, label, TOLERANCE)
    assertIterableEqualsWithTolerance(params.get._2.toArray, features.toArray, TOLERANCE)
    assertEquals(params.get._3, offset, TOLERANCE)
    assertEquals(params.get._4, weight, TOLERANCE)
  }

  /**
   * Test the extractor by base class.
   */
  @Test
  def testExtractor(): Unit = {
    val label = 1.0
    val features = DenseVector[Double](2.09, 113.0, -3.3, 150.30)
    val offset = 1.5
    val weight = 3.2
    val dataPoint = LabeledPoint(label, features, offset, weight)

    // Test the extractor
    dataPoint match {
      case LabeledPoint(l, f, o, w) =>
        assertEquals(l, label, TOLERANCE)
        assertIterableEqualsWithTolerance(f.toArray, features.toArray, TOLERANCE)
        assertEquals(o, offset, TOLERANCE)
        assertEquals(w, weight, TOLERANCE)
      case _ => throw new RuntimeException(s"extractor behavior is unexpected : [$dataPoint]")
    }
  }

  /**
   * Test computeMargin by comparing to the explicit form of calculation on specific examples.
   *
   * @param features The feature vector
   * @param offset The record offset
   * @param coef The coefficient vector
   * @param margin The expected margin
   */
  @Test(dataProvider = "getDataPoints")
  def testMargin(features: Vector[Double], offset: Double, coef: Vector[Double], margin: Double): Unit ={
    val weight = math.random
    val label = math.random
    val labeledPoint = LabeledPoint(label = label, features = features, offset = offset, weight = weight)
    val actual = labeledPoint.computeMargin(coef)
    assertEquals(actual, margin, TOLERANCE)
  }

  /**
   * Test computeMargin by comparing to the explicit form of calculation on random samples.
   *
   * @param datum Randomly generated labeled samples
   */
  @Test(dataProvider = "getRandomDataPoints")
  def testComputeMarginOnRandomlyGeneratedPoints(datum: LabeledPoint): Unit =
  {
    val coef: Vector[Double] = CommonTestUtils.generateDenseVector(PROBLEM_DIMENSION)
    val margin = datum.computeMargin(coef)
    // Compute margin explicitly
    val expectedMargin =
      (for (idx <- 0 until PROBLEM_DIMENSION) yield datum.features(idx) * coef(idx)).sum + datum.offset

    assertEquals(margin, expectedMargin, TOLERANCE, "Computed margin and expected margin don't match")
  }
}

object LabeledPointTest {
  private val PROBLEM_DIMENSION: Int = 10
  private val TOLERANCE: Double = 1.0E-9
  private val DATA_RANDOM_SEED: Int = 0
  private val TRAINING_SAMPLES = PROBLEM_DIMENSION * PROBLEM_DIMENSION
}
