package com.linkedin.photon.ml.supervised.classification

import breeze.linalg.Vector
import breeze.numerics.sigmoid
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.Regression
import org.apache.spark.rdd.RDD

/**
 * Class for the classification model trained using Logistic Regression
 * @param coefficients Model coefficients estimated for every feature
 * @param intercept Intercept (Optional)
 * @author xazhang
 * @author bdrew
 */
class LogisticRegressionModel(override val coefficients: Vector[Double], override val intercept: Option[Double])
  extends GeneralizedLinearModel(coefficients, intercept) with BinaryClassifier with Regression with Serializable {

  override def predictClassWithOffset(features: Vector[Double], offset: Double, threshold: Double = 0.5): Double = {
    predict(coefficients, intercept, features, offset, threshold)
  }

  override def predictClassAllWithOffsets(
      featuresWithOffsets: RDD[(Vector[Double], Double)],
      threshold: Double = 0.5): RDD[Double] = {

    val broadcastedModel = featuresWithOffsets.context.broadcast(this)
    featuresWithOffsets.map { case (features, offset) =>
      val coefficients = broadcastedModel.value.coefficients
      val intercept = broadcastedModel.value.intercept
      predict(coefficients, intercept, features, offset, threshold)
    }
  }

  private def predict(
      coefficients: Vector[Double],
      intercept: Option[Double],
      features: Vector[Double],
      offset: Double,
      threshold: Double): Double = {

    val score = computeMean(coefficients, intercept, features, offset)
    if (score < threshold) BinaryClassifier.negativeClassLabel else BinaryClassifier.positiveClassLabel
  }

  override protected def computeMean(
      coefficients: Vector[Double],
      intercept: Option[Double],
      features: Vector[Double],
      offset: Double): Double = {
    sigmoid(coefficients.dot(features) + intercept.getOrElse(0.0) + offset)
  }

  override def predictWithOffset(features: Vector[Double], offset: Double): Double =
    computeMean(coefficients, intercept, features, offset)

  override def predictAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double] =
    featuresWithOffsets.map(x => predictWithOffset(x._1, x._2))
}
