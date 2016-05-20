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
package com.linkedin.photon.ml

import breeze.linalg.DenseVector
import com.linkedin.photon.ml.supervised.model.CoefficientSummary
import com.linkedin.photon.ml.supervised.regression.LinearRegressionModel
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

/**
 * Unit tests for bootstrapping
 */
class BootstrapTrainingTest {

  import BootstrapTrainingTest._

  @Test
  def checkAggregateCoefficients() = {
    val toAggregate = new Array[(LinearRegressionModel, Map[String, Double])](NUM_SAMPLES)

    for (i <- -HALF_NUM_SAMPLES to HALF_NUM_SAMPLES) {
      val f = i.toDouble / HALF_NUM_SAMPLES
      val coefficients = DenseVector.ones[Double](NUM_DIMENSIONS) * f
      toAggregate(i + HALF_NUM_SAMPLES) = (new LinearRegressionModel(coefficients), Map.empty)
    }

    BootstrapTraining.aggregateCoefficientConfidenceIntervals(toAggregate).foreach(checkCoefficientSummary)
  }

  @Test
  def checkAggregateMetrics() = {
    val toAggregate = new Array[(LinearRegressionModel, Map[String, Double])](NUM_SAMPLES)
    val keys = Seq("METRIC 1", "METRIC 2", "METRIC 3")

    for (i <- -HALF_NUM_SAMPLES to HALF_NUM_SAMPLES) {
      val f = i.toDouble / HALF_NUM_SAMPLES
      val coefficients = DenseVector.ones[Double](NUM_DIMENSIONS) * f
      toAggregate(i + HALF_NUM_SAMPLES) = (new LinearRegressionModel(coefficients), keys.map(x => (x, f)).toMap)
    }

    val aggregated = BootstrapTraining.aggregateMetricsConfidenceIntervals(toAggregate)
    assertEquals(aggregated.size, keys.size, "Got expected number of metrics")
    keys.foreach(x => { assertTrue(aggregated.contains(x), s"Require metric [$x] is present in the output set") })
  }

  @DataProvider
  def invalidArgumentCases(): Array[Array[Any]] = {
    Array(
      Array("Bad samples", 0, 0.66),
      Array("Bad sample rate", 4, 0.00),
      Array("Bad sample rate", 4, 1.11)
    )
  }

  @Test(dataProvider = "invalidArgumentCases", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def checkInvalidArgumentCases(desc: String, numSamp: Int, frac: Double) = {
    BootstrapTraining.bootstrap(numSamp, frac, null, null, null, null)
  }
}

object BootstrapTrainingTest {
  val TEST_TOLERANCE = 1e-6
  val HALF_NUM_SAMPLES: Int = 15
  val NUM_SAMPLES: Int = 2 * HALF_NUM_SAMPLES + 1
  val NUM_DIMENSIONS: Int = 10

  def checkCoefficientSummary(x: CoefficientSummary): Unit = {
    assertFalse(x.getMin.isNaN || x.getMin.isInfinite, "Min must be finite")
    assertFalse(x.estimateFirstQuartile().isNaN || x.estimateFirstQuartile().isInfinite, "Q1 must be finite")
    assertFalse(x.estimateMedian().isNaN || x.estimateMedian().isInfinite, "Median must be finite")
    assertFalse(x.estimateThirdQuartile().isNaN || x.estimateThirdQuartile().isInfinite, "Q3 must be finite")
    assertFalse(x.getMax.isNaN || x.getMax.isInfinite, "Max must be finite")
    assertFalse(x.getMean.isNaN || x.getMean.isInfinite, "Mean must be finite")
    assertFalse(x.getStdDev.isNaN || x.getStdDev.isInfinite, "Standard deviation must be finite")

    assertEquals(x.getCount, NUM_SAMPLES, s"Got expected number of coefficients")
    assertEquals(x.getMax, 1.0, TEST_TOLERANCE, s"Max value matches expected")
    assertEquals(x.getMin, -1.0, TEST_TOLERANCE, s"Min value matches expected")
    assertEquals(x.getMean, 0.0, TEST_TOLERANCE, s"Mean matches expected")
    assertTrue(x.getStdDev > 0, "Standard deviation is positive")
    assertTrue(x.getMin <= x.getMean && x.getMean <= x.getMax, "Mean between min and max")
    assertTrue(x.getMin <= x.estimateFirstQuartile, "Min < Q1 estimate")
    assertTrue(x.estimateFirstQuartile <= x.estimateMedian, "Q1 <= median estimate")
    assertTrue(x.estimateMedian <= x.estimateThirdQuartile, "Median <= Q3 estimate")
    assertTrue(x.estimateThirdQuartile <= x.getMax, "Q3 <= max")
  }
}