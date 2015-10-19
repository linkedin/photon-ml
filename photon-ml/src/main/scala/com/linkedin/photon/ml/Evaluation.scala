package com.linkedin.photon.ml

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.supervised.regression.Regression
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

/**
 * A collection of evaluation metrics and functions
 * @author xazhang
 */

object Evaluation {

  /**
   * Get the binomial classification metrics. Examples include area under ROC, f-measure, etc.
   * @param dataPoints The RDD representation of labeled data points
   * @param classifier The learned classifier
   * @return The binary classification metrics
   */
  def getBinaryClassificationMetrics(dataPoints: RDD[LabeledPoint], classifier: BinaryClassifier): BinaryClassificationMetrics = {
    val broadcastedClassifier = dataPoints.context.broadcast(classifier)
    val scoreAndLabels = dataPoints.map { case LabeledPoint(label, features, offset, _) =>
      (broadcastedClassifier.value.computeScoreWithOffset(features, offset), label)
    }
    new BinaryClassificationMetrics(scoreAndLabels)
  }

  /**
   * Compute the root-mean-square error (RMSE) between the regression model's prediction and the true labels.
   * @param dataPoints The RDD representation of labeled data points
   * @param regression The learned regression model
   * @return The computed root-mean-square error (RMSE)
   */
  def computeRMSE(dataPoints: RDD[LabeledPoint], regression: Regression): Double = {
    val broadcastedRegression = dataPoints.context.broadcast(regression)
    val (nume, deno) = dataPoints.map {
      case LabeledPoint(label, features, offset, weight) =>
        val diff = broadcastedRegression.value.predict(features) - label
        (diff * diff * weight, weight)
    }.reduce((p1, p2) => (p1._1 + p2._1, p1._2 + p2._2))
    math.sqrt(nume / deno)
  }
}
