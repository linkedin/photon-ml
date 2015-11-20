package com.linkedin.photon.ml

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.metric.MetricMetadata
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, BinaryClassifier}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{PoissonRegressionModel, Regression}
import com.linkedin.photon.ml.util.Utils
import org.apache.commons.math3.special.Gamma
import org.apache.spark.Logging
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, RegressionMetrics}
import org.apache.spark.rdd.RDD

/**
 * A collection of evaluation metrics and functions
 * @author xazhang
 * @author bdrew
 */

object Evaluation extends Logging {
  val MEAN_ABSOLUTE_ERROR = "Mean absolute error"
  val MEAN_SQUARE_ERROR = "Mean square error"
  val ROOT_MEAN_SQUARE_ERROR = "Root mean square error"
  val AREA_UNDER_PRECISION_RECALL = "Area under precision/recall"
  val AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS = "Area under ROC"
  val PEAK_F1_SCORE = "Peak F1 score"
  val DATA_LOG_LIKELIHOOD = "Per-datum log likelihood"
  val AIKAKE_INFORMATION_CRITERION = "Aikake information criterion"
  val EPSILON = 1e-9

  /**
   * Assumption: model.computeMeanFunctionWithOffset is what is used to do predictions in the case of both binary
   * classification and regression; hence, it is safe to do scoring once, using this method, and then re-use to get
   * all metrics.
   *
   * @param model
   * @param dataSet
   * @return Map of (metricName &rarr; value)
   */
  def evaluate(model: GeneralizedLinearModel, dataSet: RDD[LabeledPoint]): Map[String, Double] = {
    val scoredSet = dataSet.map(x => x match {
      case LabeledPoint(label, features, offset, _) => (label, (features, offset))
    }).map(x => (x._1, model.computeMeanFunctionWithOffset(x._2._1, x._2._2))).cache

    var metrics = Map[String, Double]()

    // Compute regression facet metrics
    model match {
      case r: Regression =>
        val regressionMetrics = new RegressionMetrics(scoredSet)
        metrics ++= Map[String, Double](MEAN_ABSOLUTE_ERROR -> regressionMetrics.meanAbsoluteError,
                                        MEAN_SQUARE_ERROR -> regressionMetrics.meanSquaredError,
                                        ROOT_MEAN_SQUARE_ERROR -> regressionMetrics.rootMeanSquaredError)

      case _ =>
      // Do nothing
    }

    // Compute binary classifier metrics
    model match {
      case b: BinaryClassifier =>
        val binaryMetrics = new BinaryClassificationMetrics(scoredSet)
        metrics ++= Map[String, Double](AREA_UNDER_PRECISION_RECALL -> binaryMetrics.areaUnderPR,
                                        AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS -> binaryMetrics.areaUnderROC,
                                        PEAK_F1_SCORE -> binaryMetrics.fMeasureByThreshold.map(x => x._2).max)
      case _ =>
      // Do nothing
    }

    // Log loss
    model match {
      case p: PoissonRegressionModel =>
        metrics ++= Map[String, Double](DATA_LOG_LIKELIHOOD -> poissonRegressionLogLikelihood(dataSet, p))

      case _: LogisticRegressionModel =>
        metrics ++= Map[String, Double](DATA_LOG_LIKELIHOOD -> logisticRegressionLogLikelihood(scoredSet))

      case _ =>
      // do nothing
    }

    val aikakeInformationCriterion = metrics.get(DATA_LOG_LIKELIHOOD).map(x => {
      val n = scoredSet.count
      val logLikelihood = n * x
      val effectiveParameters = model.coefficients.activeValuesIterator.foldLeft(0)((count, coeff) => {
        if (math.abs(coeff) > 1e-9) {
          count + 1
        } else {
          count
        }
      })

      // see https://en.wikipedia.org/wiki/Akaike_information_criterion
      val baseAic = 2.0 * (effectiveParameters.toDouble - logLikelihood)
      baseAic + 2.0 * effectiveParameters * (effectiveParameters + 1) / (n - effectiveParameters - 1.0)
    })

    aikakeInformationCriterion match {
      case Some(x) => metrics ++= Map[String, Double](AIKAKE_INFORMATION_CRITERION -> x)
      case _ =>
    }

    logInfo(s"Generated metrics with keys ${metrics.keys.mkString(", ")}")

    scoredSet.unpersist(false)
    metrics
  }

  // See https://en.wikipedia.org/wiki/Poisson_regression
  private def poissonRegressionLogLikelihood(labeled: RDD[LabeledPoint], model: PoissonRegressionModel): Double = {
    val logLikelihoods = labeled.map(sample => {
      // Compute the log likelihoods
      val y = sample.label
      val intercept = if (model.hasIntercept) {
        model.intercept.get
      } else {
        0.0
      }
      val wTx = sample.computeMargin(model.coefficients) + intercept
      val numeratorLog = y * wTx - math.exp(wTx)
      val denominorLog = Gamma.logGamma(1.0 + y) // y! = Gamma(y + 1)
      numeratorLog - denominorLog
    })

    averageRDD(logLikelihoods)
  }

  // See https://en.wikipedia.org/wiki/Logistic_regression
  private def logisticRegressionLogLikelihood(scored: RDD[(Double, Double)]): Double = {
    val logLikelihood = scored.map(x => {
      val (y, p) = x
      val logP = if (p > EPSILON) math.log(p) else math.log(EPSILON)
      val log1mP = if (p > 1 - EPSILON) math.log1p(1 - EPSILON) else math.log1p(-p)
      val result =  y * logP + (1.0 - y) * log1mP
      assert(!result.isInfinite && !result.isNaN, s"y = $y, p = $p, result is not finite")
      result
    })

    averageRDD(logLikelihood)
  }

  private def averageRDD(toAverage: RDD[Double]): Double = {
    toAverage.mapPartitions(toAverage => {
      // Compute per-partion partial mean
      var count = 0
      var mean = 0.0
      toAverage.foreach(x => {
        count += 1
        mean += (x - mean) / count
      })
      Seq((count, mean)).iterator
    }).reduce((a, b) => {
      // Aggregate per-partition means
      val (countA, meanA) = a
      val (countB, meanB) = b
      val delta = meanB - meanA
      val newCount = countA + countB
      val newMean = meanA + (meanB * countB) / newCount
      (newCount, newMean)
    })._2
  }

  val sortDecreasing = new Ordering[Double]() {
    override def compare(x: Double, y: Double): Int = -x.compareTo(y)
  }

  val sortIncreasing = new Ordering[Double]() {
    override def compare(x: Double, y: Double): Int = x.compareTo(y)
  }

  val metricMetadata = Map(
    MEAN_ABSOLUTE_ERROR -> MetricMetadata(MEAN_ABSOLUTE_ERROR, "Regression metric", sortDecreasing, None),
    MEAN_SQUARE_ERROR -> MetricMetadata(MEAN_SQUARE_ERROR, "Regression metric", sortDecreasing, None),
    ROOT_MEAN_SQUARE_ERROR -> MetricMetadata(ROOT_MEAN_SQUARE_ERROR, "Regression metric", sortDecreasing, None),
    AREA_UNDER_PRECISION_RECALL -> MetricMetadata(AREA_UNDER_PRECISION_RECALL, "Binary classification metric", sortIncreasing, Some((0.0, 1.0))),
    AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS -> MetricMetadata(AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS, "Binary classification metric", sortIncreasing, Some((0.0, 1.0))),
    DATA_LOG_LIKELIHOOD -> MetricMetadata(DATA_LOG_LIKELIHOOD, "Model selection metric", sortIncreasing, None),
    AIKAKE_INFORMATION_CRITERION -> MetricMetadata(AIKAKE_INFORMATION_CRITERION, "Model selection metric", sortDecreasing, None),
    PEAK_F1_SCORE -> MetricMetadata(PEAK_F1_SCORE, "Binary classification metric", sortIncreasing, Some((0.0, 1.0))))
}
