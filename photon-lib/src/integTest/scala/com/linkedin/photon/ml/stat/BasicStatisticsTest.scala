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
package com.linkedin.photon.ml.stat

import breeze.linalg.{DenseMatrix, max => Bmax, min => Bmin, norm => Bnorm}
import breeze.stats.{MeanAndVariance, meanAndVariance}
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance
import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * Test basic statistics result.
 */
class BasicStatisticsTest extends SparkTestUtils {

  private val DELTA: Double = 1.0e-8
  private val NUM_POINTS: Int = 10
  private val NUM_FEATURES: Int = 6
  private val SEED: Int = 0

  @Test
  def testBasicStatistics(): Unit = sparkTest("testBasicStatistics") {
    val labeledPoints = drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(SEED, NUM_POINTS,
      NUM_FEATURES)
        .map(obj => new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)).toList
    val dataRdd = sc.parallelize(labeledPoints)
    val summary = BasicStatistics.getBasicStatistics(dataRdd)
    assertEquals(summary.count, NUM_POINTS.toLong)
    val allElements = labeledPoints.map(x => x.features.toArray).reduceLeft((x, y) => x ++: y)
    // A matrix with columns representing points and rows representing features.
    // The matrix is filled in column major order.
    val matrix = new DenseMatrix(NUM_FEATURES, NUM_POINTS, allElements)

    val items = for (i <- 0 until NUM_FEATURES) yield {
      // Get the i-th row and transpose to a vector. Similar to MATLAB syntax
      val vector = matrix(i, ::).t
      val oneNormL1 = Bnorm(vector, 1)
      val oneNormL2 = Bnorm(vector, 2)
      val oneMax = Bmax(vector)
      val oneMin = Bmin(vector)
      val mV: MeanAndVariance = meanAndVariance(vector)
      val oneNumNonzeros = vector.toArray.count(_ != 0).toDouble
      (oneNormL1, oneNormL2, oneMax, oneMin, mV.mean, mV.variance, oneNumNonzeros)
    }

    val normL1 = items.map(_._1)
    val normL2 = items.map(_._2)
    val meanAbs = normL1.map(_ / labeledPoints.size)
    val max = items.map(_._3)
    val min = items.map(_._4)
    val mean = items.map(_._5)
    val variance = items.map(_._6)
    val numNonzeros = items.map(_._7)

    assertIterableEqualsWithTolerance(summary.max.toArray, max, DELTA)
    assertIterableEqualsWithTolerance(summary.min.toArray, min, DELTA)
    assertIterableEqualsWithTolerance(summary.mean.toArray, mean, DELTA)
    assertIterableEqualsWithTolerance(summary.variance.toArray, variance, DELTA)
    assertIterableEqualsWithTolerance(summary.normL1.toArray, normL1, DELTA)
    assertIterableEqualsWithTolerance(summary.normL2.toArray, normL2, DELTA)
    assertIterableEqualsWithTolerance(summary.numNonzeros.toArray, numNonzeros, DELTA)
    assertIterableEqualsWithTolerance(summary.meanAbs.toArray, meanAbs, DELTA)
  }
}
