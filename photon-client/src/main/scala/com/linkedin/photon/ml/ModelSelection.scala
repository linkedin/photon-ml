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

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.metric.MetricMetadata
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.util.Logging

/**
 * A collection of functions for model selection purpose.
 */
object ModelSelection extends Logging {

  /**
   * Select the best binary classifier via AUC (Area Under ROC Curve) computed on validation data set.
   *
   * @param binaryClassifiers A map of (key -> binary classifier) to select from
   * @param validationData The validation data
   * @return A tuple of the key and the best binary classifier model according to the evaluation metric
   */
  def selectBestLinearClassifier[M <: GeneralizedLinearModel with BinaryClassifier](
      binaryClassifiers: Iterable[(Double, M)],
      validationData: RDD[LabeledPoint]): (Double, M) = {
    selectModelByKey(binaryClassifiers, validationData, Evaluation.AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS)
  }

  /**
   * Select the best linear regression model via RMSE (rooted mean square error) computed on validation data set.
   *
   * @param linearRegressionModels A map of (key -> linear regression model) to select from
   * @param validationData The validation data
   * @return A tuple of the key and the best linear regression model according to the evaluation metric
   */
  def selectBestLinearRegressionModel(
      linearRegressionModels: Iterable[(Double, LinearRegressionModel)],
      validationData: RDD[LabeledPoint]): (Double, LinearRegressionModel) = {
    selectModelByKey(linearRegressionModels, validationData, Evaluation.ROOT_MEAN_SQUARE_ERROR)
  }

  /**
   * Select the best poisson regression model via minimizing regularized log-likelihood computed on validation data set.
   *
   * @param poissonRegressionModels A map of (key -> poisson regression model) to select from
   * @param validationData The validation data
   * @return A tuple of the key and the best poisson regression model according to the evaluation metric
   */
  def selectBestPoissonRegressionModel(
      poissonRegressionModels: Iterable[(Double, PoissonRegressionModel)],
      validationData: RDD[LabeledPoint]): (Double, PoissonRegressionModel) = {
    selectModelByKey(poissonRegressionModels, validationData, Evaluation.DATA_LOG_LIKELIHOOD)
  }

  /**
   * Select the best model according to the validation data.
   *
   * @param models The models to select from
   * @param validationData The validation data to use to select the best model
   * @param metric The metric to use to select the best model
   * @tparam M The type of the model objects
   * @return The best model and the best regularization paramter corresponding to that model
   */
  private def selectModelByKey[M <: GeneralizedLinearModel](
      models:Iterable[(Double, M)],
      validationData:RDD[LabeledPoint],
      metric:String): (Double, M) = {

    val metricMetadata = Evaluation.metricMetadata.getOrElse(
      metric, MetricMetadata(metric, metric, Evaluation.sortIncreasing, None))
    val sortedByMetric = models.map(x => {
      (Evaluation.evaluate(x._2, validationData).getOrElse(metric, -1.0), x._1, x._2)
    }).toArray.sortBy(_._1)(metricMetadata.worstToBestOrdering)
    val (bestMetricValue, bestLambda, bestModel) = sortedByMetric.last
    val (worstMetricValue, worstLambda, _) = sortedByMetric.head
      logger.info(s"Selecting model with lambda = $bestLambda ($metric = $bestMetricValue) v. worst @ " +
              s"lambda = $worstLambda ($metric = $worstMetricValue)")
    (bestLambda, bestModel)
  }
}
