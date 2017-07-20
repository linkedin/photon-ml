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
package com.linkedin.photon.ml.normalization

import scala.util.Random

import breeze.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance
import com.linkedin.photon.ml.test.CommonTestUtils
import com.linkedin.photon.ml.util.VectorUtils

/**
 * Unit tests for NormalizationContext.
 */
class NormalizationContextTest {

  import NormalizationContextTest._

  @DataProvider
  def normalizationTypes(): Array[Array[Any]] = {
    Array(
      Array(NormalizationType.SCALE_WITH_STANDARD_DEVIATION, Some(STD_FACTORS), None),
      Array(NormalizationType.SCALE_WITH_MAX_MAGNITUDE, Some(MAX_MAGNITUDE_FACTORS), None),
      Array(NormalizationType.NONE, None, None),
      Array(NormalizationType.STANDARDIZATION, Some(STD_FACTORS), Some(MEAN_SHIFTS))
    )
  }

  /**
   * Test that a [[NormalizationContext]] can be correctly built from a statistical summary.
   *
   * @param normalizationType Type of normalization to build a [[NormalizationContext]] for
   * @param expectedFactors The expected scaling factors for the test [[BasicStatisticalSummary]] and
   *                        [[NormalizationType]]
   * @param expectedShifts The expected translational shifts for the test [[BasicStatisticalSummary]] and
   *                       [[NormalizationType]]
   */
  @Test(dataProvider = "normalizationTypes")
  def testBuildNormalizationContext(
    normalizationType: NormalizationType,
    expectedFactors: Option[Array[Double]],
    expectedShifts: Option[Array[Double]]): Unit = {

    val normalizationContext = NormalizationContext(normalizationType, STAT_SUMMARY, INTERCEPT_INDEX)
    val factors = normalizationContext.factors
    val shifts = normalizationContext.shifts

    (factors, expectedFactors) match {
      case (Some(_), None) | (None, Some(_)) =>
        throw new AssertionError("Calculated factors and expected factors don't match.")

      case (Some(f1), Some(f2)) =>
        assertIterableEqualsWithTolerance(f1.toArray, f2, CommonTestUtils.LOW_PRECISION_TOLERANCE)

      case (None, None) =>
    }

    (shifts, expectedShifts) match {
      case (Some(_), None) | (None, Some(_)) =>
        throw new AssertionError("Calculated shifts and expected shifts don't match.")

      case (Some(s1), Some(s2)) =>
        assertIterableEqualsWithTolerance(s1.toArray, s2, CommonTestUtils.LOW_PRECISION_TOLERANCE)

      case (None, None) =>
    }
  }

  @DataProvider
  def validInput(): Array[Array[Any]] = {
    Array(
      Array(None, None, false),
      Array(None, None, true),
      Array(Some(TEST_FACTOR), None, false),
      Array(Some(TEST_FACTOR), None, true),
      Array(None, Some(TEST_SHIFT), true),
      Array(Some(TEST_FACTOR), Some(TEST_SHIFT), true)
    )
  }

  /**
   * Verify model coefficients and transformation to original space are compatible. The margin in the original space
   * and the transformed space should be the same.
   *
   * @param factors The scaling factors
   * @param shifts The translational shifts
   * @param addIntercept Whether to include a feature representing the intercept
   */
  @Test(dataProvider = "validInput")
  def testModelToOriginalSpace(
      factors: Option[Vector[Double]],
      shifts: Option[Vector[Double]],
      addIntercept: Boolean): Unit = {

    val interceptId = if (addIntercept) Some(DIM - 1) else None
    val normalizationContext = NormalizationContext(factors, shifts, interceptId)
    val originalVector = getRandomVector(addIntercept)
    val transformedVector = transformVector(normalizationContext, originalVector)

    val transformedCoefs = getRandomVector(addIntercept)
    val originalCoefs = normalizationContext.modelToOriginalSpace(transformedCoefs)

    // Test model transformation to the original space
    Assert.assertEquals(
      originalCoefs.dot(originalVector),
      transformedCoefs.dot(transformedVector),
      CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }

  /**
   * Verify model coefficients and transformation to transformed space are compatible. The margin in the original space
   * and the transformed space should be the same.
   *
   * @param factors The scaling factors
   * @param shifts The translational shifts
   * @param addIntercept Whether to include a feature representing the intercept
   */
  @Test(dataProvider = "validInput")
  def testModelToTransformedSpace(
    factors: Option[Vector[Double]],
    shifts: Option[Vector[Double]],
    addIntercept: Boolean): Unit = {

    val interceptId = if (addIntercept) Some(DIM - 1) else None
    val normalizationContext = NormalizationContext(factors, shifts, interceptId)
    val originalVector = getRandomVector(addIntercept)
    val transformedVector = transformVector(normalizationContext, originalVector)

    val originalCoefs = getRandomVector(addIntercept)
    val transformedCoefs = normalizationContext.modelToTransformedSpace(originalCoefs)

    // Test model transformation to the transformed space
    Assert.assertEquals(
      transformedCoefs.dot(transformedVector),
      originalCoefs.dot(originalVector),
      CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }

  @DataProvider
  def invalidInput(): Array[Array[Any]] = {
    Array(
      Array(None, Some(TEST_SHIFT), false),
      Array(Some(TEST_FACTOR), Some(TEST_SHIFT), false),
      Array(Some(DenseVector(TEST_FACTOR.toArray :+ 1.0)), Some(TEST_SHIFT), true),
      Array(Some(TEST_FACTOR), Some(DenseVector(TEST_SHIFT.toArray :+ 1.0)), true)
    )
  }

  /**
   * Verify that erroneous input is rejected when attempting to create a new [[NormalizationContext]].
   *
   * @param factors The scaling factors
   * @param shifts The translational shifts
   * @param addIntercept Whether to include a feature representing the intercept
   */
  @Test(dataProvider = "invalidInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidNormalizationContext(
      factors: Option[Vector[Double]],
      shifts: Option[Vector[Double]],
      addIntercept: Boolean): Unit = {

    val interceptId = if (addIntercept) Some(DIM - 1) else None
    new NormalizationContext(factors, shifts, interceptId)
  }
}

object NormalizationContextTest {

  private val DIM = 5
  private val SEED = 1L
  private val RANDOM = new Random(SEED)
  private val SIGMA = 5.0
  private val TEST_FACTOR = DenseVector(0.5, 2.0, -0.1, -0.4, 1.0)
  private val TEST_SHIFT = DenseVector(-0.5, 0.1, 4.0, -2.0, 0.0)

  private val STAT_SUMMARY = {
    val features = Array[Vector[Double]](
      DenseVector[Double](1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
      DenseVector[Double](2.0, 0.0, -1.0, 0.0, 1.0, 0.0),
      DenseVector[Double](0.2, 0.0, 0.5, 0.0, 1.0, 0.0),
      DenseVector[Double](0.0, 10.0, 0.0, 5.0, 1.0, 0.0))

    BasicStatisticalSummary
      .calculateBasicStatistics(
        features.foldLeft(new MultivariateOnlineSummarizer()){ case (summarizer, vector) =>
          summarizer.add(VectorUtils.breezeToMllib(vector))
        })
  }
  // Expected shifts/factors from the above BasicStatisticalSummary
  private val STD_FACTORS = Array(1.09985336, 0.20591946, 1.17108008, 0.42008402, 1.0, 1.0)
  private val MAX_MAGNITUDE_FACTORS = Array(0.5, 0.1, 1.0, 0.2, 1.0, 1.0)
  private val MEAN_SHIFTS = Array(0.8, 2.75, 0.125, 1.5, 0.0, 0.0)
  private val INTERCEPT_INDEX: Option[Int] = Some(4)

  /**
   * Generate a [[DenseVector]] of values drawn randomly from a Gaussian distribution.
   *
   * @param addIntercept Whether to add an intercept term
   * @return A new [[DenseVector]]
   */
  private def getRandomVector(addIntercept: Boolean ): Vector[Double] = {
    if (addIntercept) {
      val ar = for (_ <- 0 until DIM - 1) yield RANDOM.nextGaussian() * SIGMA
      new DenseVector(ar.toArray :+ 1.0)
    } else {
      val ar = for (_ <- 0 until DIM) yield RANDOM.nextGaussian() * SIGMA
      new DenseVector(ar.toArray)
    }
  }

  /**
   * For testing purpose only. This is not designed to be efficient. This method transforms a vector from the original
   * space to a normalized space.
   *
   * @param input Input vector
   * @return Transformed vector
   */
  def transformVector(normalizationContext: NormalizationContext, input: Vector[Double]): Vector[Double] = {
    (normalizationContext.factors, normalizationContext.shifts) match {
      case (Some(fs), Some(ss)) =>
        require(fs.size == input.size, "Vector size and the scaling factor size are different.")
        (input - ss) :* fs
      case (Some(fs), None) =>
        require(fs.size == input.size, "Vector size and the scaling factor size are different.")
        input :* fs
      case (None, Some(ss)) =>
        require(ss.size == input.size, "Vector size and the scaling factor size are different.")
        input - ss
      case (None, None) =>
        input
    }
  }
}
