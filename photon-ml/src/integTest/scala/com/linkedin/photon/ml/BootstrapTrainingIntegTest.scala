package com.linkedin.photon.ml

import breeze.linalg.DenseVector
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, CoefficientSummary}
import com.linkedin.photon.ml.supervised.regression.LinearRegressionModel
import com.linkedin.photon.ml.test.SparkTestUtils
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

/**
 * Integration tests for bootstrapping. Most of the heavy lifting has already been done in the unit tests
 */
class BootstrapTrainingIntegTest extends SparkTestUtils {

  import org.testng.Assert._

  val lambdas: List[Double] = List(0.01, 0.1, 1.0)
  val numWorkers: Int = Math.max(1, Runtime.getRuntime.availableProcessors / 2)
  val samplePct = 0.01
  val seed = 0L
  val numSamples = 100

  def regressionModelFitFunction(coefficient: Double, lambdas: Seq[Double]): (RDD[LabeledPoint], Map[Double, LinearRegressionModel]) => List[(Double, LinearRegressionModel)] = {
    (x: RDD[LabeledPoint], y: Map[Double, LinearRegressionModel]) => {
      lambdas.map(l => (l, new LinearRegressionModel(DenseVector.ones[Double](BootstrapTrainingTest.NUM_DIMENSIONS) * coefficient, None))).toList
    }
  }

  /**
   * Sanity check that the bootstrapping mechanics appear to work before we attempt to do integration tests with
   * "real" aggregation operations and data sets
   */
  @Test
  def checkBootstrapHappyPathRegressionDummyAggregates(): Unit = sparkTest("checkBootstrapHappyPathDummyAggregates") {
    val identity = (x: Seq[(LinearRegressionModel, Map[String, Double])]) => {
      x
    }
    val identityKey: String = "identity"
    val aggregations: Map[String, Seq[(LinearRegressionModel, Map[String, Double])] => Any] = Map(identityKey -> identity)

    // Generate an empty RDD (model fitting is mocked out but we need a "real" instance for the sampling to work)
    val data: RDD[LabeledPoint] = sc.parallelize(drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(seed.toInt, numSamples, BootstrapTrainingTest.NUM_DIMENSIONS).toSeq).map(x => new LabeledPoint(x._1, x._2))

    val result: Map[Double, Map[String, Any]] = BootstrapTraining.bootstrap[LinearRegressionModel](
      BootstrapTrainingTest.NUM_SAMPLES,
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
                case m: TraversableOnce[(LinearRegressionModel, Map[String, Double])] => assertEquals(m.size, BootstrapTrainingTest.NUM_SAMPLES, "Number of bootstrapped models matches expected")
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
  @Test
  def checkBootstrapHappyPathRealAggregates(): Unit = sparkTest("checkBootstrapHappyPathRealAggregates") {
    // Return a different model each time fitFunction is called
    var count: Int = -BootstrapTrainingTest.HALF_NUM_SAMPLES
    val fitFunction = (x: RDD[LabeledPoint], y: Map[Double, LinearRegressionModel]) => {
      val value = count / BootstrapTrainingTest.HALF_NUM_SAMPLES.toDouble
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
            BootstrapTrainingTest.checkCoefficientSummary(c)
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

    val data: RDD[LabeledPoint] = sc.parallelize(drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(seed.toInt, numSamples, BootstrapTrainingTest.NUM_DIMENSIONS).toSeq).map(x => new LabeledPoint(x._1, x._2)).coalesce(4)

    val aggregates: Map[Double, Map[String, Any]] = BootstrapTraining.bootstrap[LinearRegressionModel](
      BootstrapTrainingTest.NUM_SAMPLES,
      samplePct,
      Map[Double, LinearRegressionModel](),
      fitFunction,
      aggregations,
      data)
  }
}


