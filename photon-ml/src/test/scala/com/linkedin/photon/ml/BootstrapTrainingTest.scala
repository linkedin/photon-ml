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
  import org.testng.Assert._
  import java.lang.{Long => JLong}
  import java.lang.{Integer => JInt}
  import java.lang.{Double => JDouble}

  @Test
  def checkAggregateCoefficientsHappyPathWithIntercept() = {
    val toAggregate: Array[(LinearRegressionModel, Map[String, Double])] = new Array[(LinearRegressionModel, Map[String, Double])](NUM_SAMPLES)

    for (i <- -HALF_NUM_SAMPLES to HALF_NUM_SAMPLES) {
      val f = i.toDouble / HALF_NUM_SAMPLES
      val coefficients = DenseVector.ones[Double](NUM_DIMENSIONS) * f
      val intercept = Option(f)
      toAggregate(i + HALF_NUM_SAMPLES) = (new LinearRegressionModel(coefficients, intercept), Map.empty)
    }

    val result = BootstrapTraining.aggregateCoefficientConfidenceIntervals(toAggregate)

    result match {
      case (coeffSum: Array[CoefficientSummary], Some(intSum)) =>
        coeffSum.foreach(x => {
          checkCoefficientSummary(x)
        })

        checkCoefficientSummary(intSum)
      case _ => fail("Should have computed a summary for intercept")
    }
  }

  @Test
  def checkAggregateCoefficientsHappyPathWithoutIntercept() = {
    val toAggregate: Array[(LinearRegressionModel, Map[String, Double])] = new Array[(LinearRegressionModel, Map[String, Double])](NUM_SAMPLES)

    for (i <- -HALF_NUM_SAMPLES to HALF_NUM_SAMPLES) {
      val f = i.toDouble / HALF_NUM_SAMPLES
      val coefficients = DenseVector.ones[Double](NUM_DIMENSIONS) * f
      val intercept = None
      toAggregate(i + HALF_NUM_SAMPLES) = (new LinearRegressionModel(coefficients, intercept), Map.empty)
    }

    val result = BootstrapTraining.aggregateCoefficientConfidenceIntervals(toAggregate)

    result match {
      case (coeffSum: Array[CoefficientSummary], None) =>
        coeffSum.foreach(x => {
          checkCoefficientSummary(x)
        })

      case _ => fail("Should have computed a summary for intercept")
    }
  }

  @Test
  def checkAggregateMetrics() = {
    val toAggregate: Array[(LinearRegressionModel, Map[String, Double])] = new Array[(LinearRegressionModel, Map[String, Double])](NUM_SAMPLES)
    val keys = Seq("METRIC 1", "METRIC 2", "METRIC 3")

    for (i <- -HALF_NUM_SAMPLES to HALF_NUM_SAMPLES) {
      val f = i.toDouble / HALF_NUM_SAMPLES
      val coefficients = DenseVector.ones[Double](NUM_DIMENSIONS) * f
      val intercept = None
      toAggregate(i + HALF_NUM_SAMPLES) = (new LinearRegressionModel(coefficients, intercept), keys.map(x => (x, f)).toMap)
    }

    val aggregated = BootstrapTraining.aggregateMetricsConfidenceIntervals(toAggregate)
    assertEquals(aggregated.size, keys.size, "Got expected number of metrics")
    keys.map(x => {
      assertTrue(aggregated.contains(x), s"Require metric [$x] is present in the output set")
    })
  }

  @DataProvider
  def invalidArgumentCases(): Array[Array[Any]] = {
    Array(
      Array("Bad concurrency", 0, 4, 0.66),
      Array("Bad samples", 1, 0, 0.66),
      Array("Bad sample rate", 1, 4, 0.00),
      Array("Bad sample rate", 1, 4, 1.11)
    )
  }

  @Test(dataProvider = "invalidArgumentCases", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def checkInvalidArgumentCases(desc: String, numConc: Int, numSamp: Int, frac: Double) = {
    BootstrapTraining.bootstrap(numConc, numSamp, 0L, frac, null, null, null)
  }
}

object BootstrapTrainingTest {
  val TEST_TOLERANCE = 1e-6
  val HALF_NUM_SAMPLES: Int = 15
  val NUM_SAMPLES: Int = 2 * HALF_NUM_SAMPLES + 1
  val NUM_DIMENSIONS: Int = 10

  def checkCoefficientSummary(x: CoefficientSummary): Unit = {
    assertFalse(x.getMin.isNaN || x.getMin.isInfinite, "Min must be finite")
    assertFalse(x.estimateFirstQuartile.isNaN || x.estimateFirstQuartile.isInfinite, "Q1 must be finite")
    assertFalse(x.estimateMedian.isNaN || x.estimateMedian.isInfinite, "Median must be finite")
    assertFalse(x.estimateThirdQuartile.isNaN || x.estimateThirdQuartile.isInfinite, "Q3 must be finite")
    assertFalse(x.getMax.isNaN || x.getMax.isInfinite, "Max must be finite")
    assertFalse(x.getMean.isNaN || x.getMean.isInfinite, "Mean must be finite")
    assertFalse(x.getStdDev.isNaN || x.getStdDev.isInfinite, "Standard deviation must be finite")

    assertEquals(x.getCount(), NUM_SAMPLES, s"Got expected number of coefficients")
    assertEquals(x.getMax(), 1.0, TEST_TOLERANCE, s"Max value matches expected")
    assertEquals(x.getMin(), -1.0, TEST_TOLERANCE, s"Min value matches expected")
    assertEquals(x.getMean(), 0.0, TEST_TOLERANCE, s"Mean matches expected")
    assertTrue(x.getStdDev() > 0, "Standard deviation is positive")
    assertTrue(x.getMin <= x.getMean && x.getMean <= x.getMax, "Mean between min and max")
    assertTrue(x.getMin <= x.estimateFirstQuartile, "Min < Q1 estimate")
    assertTrue(x.estimateFirstQuartile <= x.estimateMedian, "Q1 <= median estimate")
    assertTrue(x.estimateMedian <= x.estimateThirdQuartile, "Median <= Q3 estimate")
    assertTrue(x.estimateThirdQuartile <= x.getMax, "Q3 <= max")
  }
}