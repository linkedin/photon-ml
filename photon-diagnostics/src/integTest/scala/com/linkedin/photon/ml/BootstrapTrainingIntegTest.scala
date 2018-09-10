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
package com.linkedin.photon.ml

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.model.CoefficientSummary
import com.linkedin.photon.ml.supervised.regression.LinearRegressionModel
import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * Integration tests for bootstrapping. Most of the heavy lifting has already been done in the unit tests
 */
class BootstrapTrainingIntegTest extends SparkTestUtils {

  import org.testng.Assert._

  private val lambdas: List[Double] = List(0.01, 0.1, 1.0)
  private val samplePct = 0.01
  private val seed = 0L
  private val numSamples = 100
  private val halfNumSamples = numSamples / 2
  private val numDimensions = 10
  private val testTolerance = 1e-6

  def regressionModelFitFunction(coefficient: Double, lambdas: Seq[Double])
    : (RDD[LabeledPoint], Map[Double, LinearRegressionModel]) => List[(Double, LinearRegressionModel)] = {

    (x: RDD[LabeledPoint], y: Map[Double, LinearRegressionModel]) => {
      lambdas.map(l => {
        (l, new LinearRegressionModel(
          Coefficients(DenseVector.ones[Double](numDimensions) * coefficient)))
      })
        .toList
    }
  }

  /**
   * Sanity check that the bootstrapping mechanics appear to work before we attempt to do integration tests with
   * "real" aggregation operations and datasets
   */
  @Test
  def checkBootstrapHappyPathRegressionDummyAggregates(): Unit = sparkTest("checkBootstrapHappyPathDummyAggregates") {
    val identity = (x: Seq[(LinearRegressionModel, Map[String, Double])]) => x
    val identityKey: String = "identity"
    val aggregations: Map[String, Seq[(LinearRegressionModel, Map[String, Double])] => Any] = Map(identityKey -> identity)

    // Generate an empty RDD (model fitting is mocked out but we need a "real" instance for the sampling to work)
    val data: RDD[LabeledPoint] = sc.parallelize(
      drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(
        seed.toInt,
        numSamples,
        numDimensions)
        .toSeq)
      .map(x => new LabeledPoint(x._1, x._2))

    val result: Map[Double, Map[String, Any]] = BootstrapTraining.bootstrap[LinearRegressionModel](
      numSamples,
      samplePct,
      Map[Double, LinearRegressionModel](),
      regressionModelFitFunction(0, lambdas),
      aggregations,
      data)

    // Verify that we got the expected results
    assertEquals(result.size, lambdas.size, "Result has expected number of keys")
    lambdas.foreach(x => {
      result.get(x) match {
        case Some(aggregates) =>
          aggregates.get(identityKey) match {
            case Some(models) =>
              models match {
                case m: TraversableOnce[(LinearRegressionModel, Map[String, Double])] =>
                  assertEquals(
                    m.size,
                    numSamples,
                    "Number of bootstrapped models matches expected")
                case _ => fail(f"Found aggregate for lambda=[$x%.04f] and name [$identityKey] with unexpected type")
              }
            case None =>
              fail(f"Aggregate [$identityKey] appears to be missing")
          }

        case None =>
          fail(f"Result is missing aggregates for lambda = [$x%.04f]")

        case _ =>
          fail(f"Result has aggregates for lambda = [$x%.04f] with unexpected type")
      }
    })
  }

  /**
   * "Real" integration test where we hook in all the aggregation operations and sanity check their output
   */
  //TODO: Add sanity checks
  @Test
  def checkBootstrapHappyPathRealAggregates(): Unit = sparkTest("checkBootstrapHappyPathRealAggregates") {
    // Return a different model each time fitFunction is called
    var count: Int = -halfNumSamples
    val fitFunction = (x: RDD[LabeledPoint], y: Map[Double, LinearRegressionModel]) => {
      val value = count / halfNumSamples.toDouble
      count += 1
      val fn = regressionModelFitFunction(value, lambdas)
      fn(x, y)
    }

    val confidenceIntervalsKey = "confidenceIntervalEstimate"
    val metricsIntervalsKey = "metricsIntervalEstimate"
    val aggregations: Map[String, Seq[(LinearRegressionModel, Map[String, Double])] => Any] = Map(
      confidenceIntervalsKey -> BootstrapTraining.aggregateCoefficientConfidenceIntervals,
      metricsIntervalsKey -> BootstrapTraining.aggregateMetricsConfidenceIntervals
    )

    val validateConfidenceIntervals: Any => Unit = x => {
      x match {
        case (coeff: Array[CoefficientSummary], intercept: Option[CoefficientSummary]) =>
          coeff.foreach(c => {
            checkCoefficientSummary(c)
          })
          intercept match {
            case Some(_) => fail("Intercept should not have been computed")
            case None =>
          }
        case _ =>
          fail(s"Aggregate for $confidenceIntervalsKey is of unexpected type")
      }
    }: Unit

    val validateMetricsIntervals: Any => Unit = x => {
      x match {
        case m: Map[String, CoefficientSummary] =>
        case _ =>
          fail(s"Aggregate for $metricsIntervalsKey is of unexpected type")
      }
    }: Unit

    val aggregationValidators: Map[String, Any => Unit] = Map(
      confidenceIntervalsKey -> validateConfidenceIntervals,
      metricsIntervalsKey -> validateMetricsIntervals
    )

    val data: RDD[LabeledPoint] = sc.parallelize(
      drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(
        seed.toInt,
        numSamples,
        numDimensions)
          .toSeq)
        .map(x => new LabeledPoint(x._1, x._2))
        .coalesce(4)

    val aggregates: Map[Double, Map[String, Any]] = BootstrapTraining.bootstrap[LinearRegressionModel](
      numSamples,
      samplePct,
      Map[Double, LinearRegressionModel](),
      fitFunction,
      aggregations,
      data)
  }

  def checkCoefficientSummary(x: CoefficientSummary): Unit = {
    assertFalse(x.getMin.isNaN || x.getMin.isInfinite, "Min must be finite")
    assertFalse(x.estimateFirstQuartile().isNaN || x.estimateFirstQuartile().isInfinite, "Q1 must be finite")
    assertFalse(x.estimateMedian().isNaN || x.estimateMedian().isInfinite, "Median must be finite")
    assertFalse(x.estimateThirdQuartile().isNaN || x.estimateThirdQuartile().isInfinite, "Q3 must be finite")
    assertFalse(x.getMax.isNaN || x.getMax.isInfinite, "Max must be finite")
    assertFalse(x.getMean.isNaN || x.getMean.isInfinite, "Mean must be finite")
    assertFalse(x.getStdDev.isNaN || x.getStdDev.isInfinite, "Standard deviation must be finite")
    assertEquals(x.getCount, numSamples, s"Got expected number of coefficients")
    assertEquals(x.getMax, 1.0, testTolerance, s"Max value matches expected")
    assertEquals(x.getMin, -1.0, testTolerance, s"Min value matches expected")
    assertEquals(x.getMean, 0.0, testTolerance, s"Mean matches expected")
    assertTrue(x.getStdDev > 0, "Standard deviation is positive")
    assertTrue(x.getMin <= x.getMean && x.getMean <= x.getMax, "Mean between min and max")
    assertTrue(x.getMin <= x.estimateFirstQuartile, "Min < Q1 estimate")
    assertTrue(x.estimateFirstQuartile <= x.estimateMedian, "Q1 <= median estimate")
    assertTrue(x.estimateMedian <= x.estimateThirdQuartile, "Median <= Q3 estimate")
    assertTrue(x.estimateThirdQuartile <= x.getMax, "Q3 <= max")
  }
}
