package com.linkedin.photon.ml

import breeze.optimize.L2Regularization
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.PoissonLossFunction
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import org.apache.spark.rdd.RDD

/**
 * A collection of functions for model selection purpose
 * @author xazhang
 * @author dpeng
 */
object ModelSelection
{

  /**
   * Select the best binary classifier via AUC (Area Under ROC Curve) computed on validating data set
   * @param binaryClassifiers A map of (key -> binary classifier) to select from
   * @param validatingData The validating data
   * @return A tuple of the key and the best binary classifier model according to the evaluation metric
   */
  def selectBestBinaryClassifier(binaryClassifiers: Iterable[(Double, BinaryClassifier)], validatingData: RDD[LabeledPoint]): (Double, BinaryClassifier) =
  {
    binaryClassifiers.map
            { case (weight, classifier) =>
              ((weight, classifier), Evaluation.getBinaryClassificationMetrics(validatingData, classifier).areaUnderROC())
            }.toArray.sortBy(_._2).last._1
  }

  /**
   * Select the best linear regression model via RMSE (rooted mean square error) computed on validating data set
   * @param linearRegressionModels A map of (key -> linear regression model) to select from
   * @param validatingData The validating data
   * @return A tuple of the key and the best linear regression model according to the evaluation metric
   */
  def selectBestLinearRegressionModel(linearRegressionModels: Iterable[(Double, LinearRegressionModel)], validatingData: RDD[LabeledPoint]): (Double, LinearRegressionModel) =
  {
    linearRegressionModels.map
            { case (weight, regression) =>
              ((weight, regression), Evaluation.computeRMSE(validatingData, regression))
            }.toArray.sortBy(_._2).head._1
  }

  /**
   * Select the best poisson regression model via minimizing regularized log-likelihood computed on validating data set
   * @param poissonRegressionModels A map of (key -> poisson regression model) to select from
   * @param validatingData The validating data
   * @return A tuple of the key and the best poisson regression model according to the evaluation metric
   */
  def selectBestPoissonRegressionModel(poissonRegressionModels: Iterable[(Double, PoissonRegressionModel)], validatingData: RDD[LabeledPoint]): (Double, PoissonRegressionModel) =
  {
    poissonRegressionModels.map
            { case (weight, poissonRegression) =>
              ((weight, poissonRegression), Evaluation.computeRMSE(validatingData, poissonRegression))
            }.toArray.sortBy(_._2).head._1
  }
}

