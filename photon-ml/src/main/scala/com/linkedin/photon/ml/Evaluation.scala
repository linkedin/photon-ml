package com.linkedin.photon.ml

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.metric.MetricMetadata
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.Regression
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, RegressionMetrics}
import org.apache.spark.rdd.RDD

/**
 * A collection of evaluation metrics and functions
 * @author xazhang
 */

object Evaluation {
  val MEAN_ABSOLUTE_ERROR = "Mean absolute error"
  val MEAN_SQUARE_ERROR = "Mean square error"
  val ROOT_MEAN_SQUARE_ERROR = "Root mean square error"
  val AREA_UNDER_PRECISION_RECALL = "Area under precision/recall"
  val AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS = "Area under ROC"
  val PEAK_F1_SCORE = "Peak F1 score"

  /**
   * Assumption: model.computeMeanFunctionWithOffset is what is used to do predictions in the case of both binary
   * classification and regression; hence, it is safe to do scoring once, using this method, and then re-use to get
   * all metrics.
   *
   * @param model
   * @param dataSet
   * @return Map of (metricName &rarr; value)
   */
  def evaluate(model:GeneralizedLinearModel, dataSet:RDD[LabeledPoint]):Map[String, Double] = {
    val scoredSet = dataSet.map(x => x match {
      case LabeledPoint(label, features, offset, _) => (label, (features, offset))
    }).map(x => (x._1, model.computeMeanFunctionWithOffset(x._2._1, x._2._2)))

    var metrics = Map[String, Double]()

    // Compute regression facet metrics
    model match {
      case r:Regression =>
        val regressionMetrics = new RegressionMetrics(scoredSet)
        metrics ++= Map[String, Double](MEAN_ABSOLUTE_ERROR -> regressionMetrics.meanAbsoluteError,
                                        MEAN_SQUARE_ERROR -> regressionMetrics.meanSquaredError,
                                        ROOT_MEAN_SQUARE_ERROR -> regressionMetrics.rootMeanSquaredError)

      case _ =>
        // Do nothing
    }

    // Compute binary classifier metrics
    model match {
      case b:BinaryClassifier =>
        val binaryMetrics = new BinaryClassificationMetrics(scoredSet)
        metrics ++= Map[String, Double](AREA_UNDER_PRECISION_RECALL -> binaryMetrics.areaUnderPR,
                                        AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS -> binaryMetrics.areaUnderROC,
                                        PEAK_F1_SCORE -> binaryMetrics.fMeasureByThreshold.map(x => x._2).max)
      case _ =>
        // Do nothing
    }

    // Additional metrics (e.g. loss, log loss) go here

    metrics
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
    PEAK_F1_SCORE -> MetricMetadata(PEAK_F1_SCORE, "Binary classification metric", sortIncreasing, Some((0.0, 1.0))))
}
