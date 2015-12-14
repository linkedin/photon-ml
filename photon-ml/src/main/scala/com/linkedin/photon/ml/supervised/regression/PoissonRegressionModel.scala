package com.linkedin.photon.ml.supervised.regression

import breeze.linalg.Vector
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

/**
 * Class for the classification model trained using Poisson Regression
 * @param coefficients Weights estimated for every feature
 * @param intercept Intercept computed for this model (Option)
 * @author asaha
 */
class PoissonRegressionModel(
    override val coefficients: Vector[Double],
    override val intercept: Option[Double])
  extends GeneralizedLinearModel(coefficients, intercept)
  with Regression
  with Serializable {

  override def predictWithOffset(features: Vector[Double], offset: Double): Double = {
    computeMeanFunctionWithOffset(features, offset)
  }

  override def predictAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double] = {
    computeMeanFunctionsWithOffsets(featuresWithOffsets)
  }

  /**
   * Compute the mean of the Poisson regression model
   * @param coefficients the estimated features' coefficients
   * @param intercept the estimated model intercept
   * @param features the input data point's feature
   * @param offset the input data point's offset
   * @return
   */
  override protected def computeMean(
      coefficients: Vector[Double],
      intercept: Option[Double],
      features: Vector[Double],
      offset: Double): Double = {
    math.exp(coefficients.dot(features) + intercept.getOrElse(0.0) + offset)
  }
}
