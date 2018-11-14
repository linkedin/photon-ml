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
package com.linkedin.photon.ml.evaluation

import org.apache.commons.math3.special.Gamma
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, RegressionMetrics}
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.evaluation.metric.MetricMetadata
import com.linkedin.photon.ml.supervised.classification.{BinaryClassifier, LogisticRegressionModel}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{PoissonRegressionModel, Regression}
import com.linkedin.photon.ml.util.Logging

/**
 * A collection of evaluation metrics and functions
 */
object Evaluation extends Logging {

  val MEAN_ABSOLUTE_ERROR = "Mean absolute error"
  val MEAN_SQUARE_ERROR = "Mean square error"
  val ROOT_MEAN_SQUARE_ERROR = "Root mean square error"
  val AREA_UNDER_PRECISION_RECALL = "Area under precision/recall"
  val AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS = "Area under ROC"
  val PEAK_F1_SCORE = "Peak F1 score"
  val DATA_LOG_LIKELIHOOD = "Per-datum log likelihood"
  val AKAIKE_INFORMATION_CRITERION = "Akaike information criterion"
  val EPSILON = 1e-9

  type MetricsMap = Map[String, Double]
  private def MetricsMap() = Map[String, Double]()

  /**
   * Assumption: model.computeMeanFunctionWithOffset is what is used to do predictions in the case of both binary
   * classification and regression; hence, it is safe to do scoring once, using this method, and then re-use to get
   * all metrics.
   *
   * @param model The GLM model to be evaluated
   * @param dataset The dataset used to evaluate the GLM model
   * @return Map of (metricName &rarr; value)
   */
  def evaluate(model: GeneralizedLinearModel, dataset: RDD[LabeledPoint]): MetricsMap = {
    val broadcastModel = dataset.sparkContext.broadcast(model)
    val scoreAndLabel = dataset
      .map(labeledPoint =>
        (broadcastModel.value.computeMeanFunctionWithOffset(labeledPoint.features, labeledPoint.offset),
          labeledPoint.label))
      .cache()
    broadcastModel.unpersist()

    var metrics = MetricsMap()

    // Compute regression facet metrics
    model match {
      case _: Regression =>
        val regressionMetrics = new RegressionMetrics(scoreAndLabel)
        metrics ++= List(
          (MEAN_ABSOLUTE_ERROR, regressionMetrics.meanAbsoluteError),
          (MEAN_SQUARE_ERROR, regressionMetrics.meanSquaredError),
          (ROOT_MEAN_SQUARE_ERROR, regressionMetrics.rootMeanSquaredError))

      case _ =>
        // Do nothing
    }

    // Compute binary classifier metrics
    model match {
      case _: BinaryClassifier =>
        val binaryMetrics = new BinaryClassificationMetrics(scoreAndLabel)
        metrics ++= List(
          (AREA_UNDER_PRECISION_RECALL, binaryMetrics.areaUnderPR),
          (AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS, binaryMetrics.areaUnderROC),
          (PEAK_F1_SCORE, binaryMetrics.fMeasureByThreshold().map(x => x._2).max))

      case _ =>
        // Do nothing
    }

    // Log loss
    model match {
      case p: PoissonRegressionModel =>
        metrics += ((DATA_LOG_LIKELIHOOD, poissonRegressionLogLikelihood(dataset, p)))

      case _: LogisticRegressionModel =>
        metrics += ((DATA_LOG_LIKELIHOOD, logisticRegressionLogLikelihood(scoreAndLabel)))

      case _ =>
        // Do nothing
    }

    val akaikeInformationCriterion = metrics.get(DATA_LOG_LIKELIHOOD).map(x => {
      val n = scoreAndLabel.count()
      val logLikelihood = n * x
      val effectiveParameters = model.coefficients.means.activeValuesIterator.foldLeft(0)((count, coeff) => {
        if (math.abs(coeff) > 1e-9) {
          count + 1
        } else {
          count
        }
      })

      // See https://en.wikipedia.org/wiki/Akaike_information_criterion
      val baseAic = 2.0 * (effectiveParameters.toDouble - logLikelihood)
      baseAic + 2.0 * effectiveParameters * (effectiveParameters + 1) / (n - effectiveParameters - 1.0)
    })

    akaikeInformationCriterion match {
      case Some(x) => metrics += ((AKAIKE_INFORMATION_CRITERION, x))
      case _ =>
    }

    logger.info(s"Generated metrics with keys ${metrics.keys.mkString(", ")}")

    scoreAndLabel.unpersist(blocking = false)
    metrics
  }

  // See https://en.wikipedia.org/wiki/Poisson_regression
  private def poissonRegressionLogLikelihood(labeled: RDD[LabeledPoint], model: PoissonRegressionModel): Double = {

    val logLikelihoods = labeled.map { sample =>
      // Compute the log likelihoods
      val y = sample.label
      val wTx = sample.computeMargin(model.coefficients.means)
      val numeratorLog = y * wTx - math.exp(wTx)
      val denominatorLog = Gamma.logGamma(1.0 + y) // y! = Gamma(y + 1)

      (numeratorLog - denominatorLog, 1)
    }

    averageLogLikelihoodRDD(logLikelihoods)
  }

  // See https://en.wikipedia.org/wiki/Logistic_regression
  private def logisticRegressionLogLikelihood(scoreAndLabel: RDD[(Double, Double)]): Double = {

    val logLikelihoods = scoreAndLabel.map { case (score, label) =>
      val logP = if (score > EPSILON) math.log(score) else math.log(EPSILON)
      val log1mP = if (score > 1 - EPSILON) math.log(EPSILON) else math.log1p(-score)
      val result = label * logP + (1.0 - label) * log1mP

      assert(!result.isInfinite && !result.isNaN, s"label = $label, score = $score, result is not finite")

      (result, 1)
    }

    averageLogLikelihoodRDD(logLikelihoods)
  }

  private def averageLogLikelihoodRDD(logLikelihoods: RDD[(Double, Int)]): Double = {

    val (logLikelihood, count) = logLikelihoods.reduce { (a, b) =>
      val (logLikelihoodA, countA) = a
      val (logLikelihoodB, countB) = b

      (logLikelihoodA + logLikelihoodB, countA + countB)
    }

    logLikelihood / count.toDouble
  }

  val sortDecreasing = new Ordering[Double]() {
    override def compare(x: Double, y: Double): Int = -x.compareTo(y)
  }

  val sortIncreasing = new Ordering[Double]() {
    override def compare(x: Double, y: Double): Int = x.compareTo(y)
  }

  val metricMetadata: Map[String, MetricMetadata] = List(
    (MEAN_ABSOLUTE_ERROR, MetricMetadata(MEAN_ABSOLUTE_ERROR, "Regression metric", sortDecreasing, None)),
    (MEAN_SQUARE_ERROR, MetricMetadata(MEAN_SQUARE_ERROR, "Regression metric", sortDecreasing, None)),
    (ROOT_MEAN_SQUARE_ERROR, MetricMetadata(ROOT_MEAN_SQUARE_ERROR, "Regression metric", sortDecreasing, None)),
    (AREA_UNDER_PRECISION_RECALL, MetricMetadata(
      AREA_UNDER_PRECISION_RECALL, "Binary classification metric", sortIncreasing, Some((0.0, 1.0)))),
    (AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS, MetricMetadata(
      AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS, "Binary classification metric", sortIncreasing, Some((0.0, 1.0)))),
    (DATA_LOG_LIKELIHOOD, MetricMetadata(DATA_LOG_LIKELIHOOD, "Model selection metric", sortIncreasing, None)),
    (AKAIKE_INFORMATION_CRITERION, MetricMetadata(
      AKAIKE_INFORMATION_CRITERION, "Model selection metric", sortDecreasing, None)),
    (PEAK_F1_SCORE, MetricMetadata(PEAK_F1_SCORE, "Binary classification metric", sortIncreasing, Some((0.0, 1.0)))))
    .toMap
}
