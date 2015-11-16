package com.linkedin.photon.ml

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.metric.MetricMetadata
import com.linkedin.photon.ml.supervised.classification.{BinaryClassifier, LogisticRegressionModel}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

/**
 * A collection of functions for model selection purpose
 * @author xazhang
 * @author dpeng
 * @author bdrew
 */
object ModelSelection extends Logging
{

  /**
   * Select the best binary classifier via AUC (Area Under ROC Curve) computed on validating data set
   * @param binaryClassifiers A map of (key -> binary classifier) to select from
   * @param validatingData The validating data
   * @return A tuple of the key and the best binary classifier model according to the evaluation metric
   */
  def selectBestLinearClassifier[M <: GeneralizedLinearModel with BinaryClassifier](binaryClassifiers: Iterable[(Double, M)], validatingData: RDD[LabeledPoint]): (Double, M) = {
    selectModelByKey(binaryClassifiers, validatingData, Evaluation.AREA_UNDER_RECEIVER_OPERATOR_CHARACTERISTICS)
  }

  /**
   * Select the best linear regression model via RMSE (rooted mean square error) computed on validating data set
   * @param linearRegressionModels A map of (key -> linear regression model) to select from
   * @param validatingData The validating data
   * @return A tuple of the key and the best linear regression model according to the evaluation metric
   */
  def selectBestLinearRegressionModel(linearRegressionModels: Iterable[(Double, LinearRegressionModel)], validatingData: RDD[LabeledPoint]): (Double, LinearRegressionModel) =
  {
    selectModelByKey(linearRegressionModels, validatingData, Evaluation.ROOT_MEAN_SQUARE_ERROR)
  }

  /**
   * Select the best poisson regression model via minimizing regularized log-likelihood computed on validating data set
   * @param poissonRegressionModels A map of (key -> poisson regression model) to select from
   * @param validatingData The validating data
   * @return A tuple of the key and the best poisson regression model according to the evaluation metric
   */
  def selectBestPoissonRegressionModel(poissonRegressionModels: Iterable[(Double, PoissonRegressionModel)], validatingData: RDD[LabeledPoint]): (Double, PoissonRegressionModel) =
  {
    selectModelByKey(poissonRegressionModels, validatingData, Evaluation.DATA_LOG_LIKELIHOOD)
  }


  private def selectModelByKey[M <: GeneralizedLinearModel](models:Iterable[(Double, M)], validatingData:RDD[LabeledPoint], metric:String): (Double, M) = {
    val metricMetadata = Evaluation.metricMetadata.getOrElse(metric, MetricMetadata(metric, metric, Evaluation.sortIncreasing, None))
    val sortedByMetric = models.map(x => {
      (Evaluation.evaluate(x._2, validatingData).getOrElse(metric, -1.0), x._1, x._2)
    }).toArray.sortBy(_._1)(metricMetadata.worstToBestOrdering)
    val (bestMetricValue, bestLambda, bestModel) = sortedByMetric.last
    val (worstMetricValue, worstLambda, worstModel) = sortedByMetric.head
      logInfo(s"Selecting model with lambda = $bestLambda ($metric = $bestMetricValue) v. worst @ lambda = $worstLambda ($metric = $worstMetricValue)")
    (bestLambda, bestModel)
  }
}

