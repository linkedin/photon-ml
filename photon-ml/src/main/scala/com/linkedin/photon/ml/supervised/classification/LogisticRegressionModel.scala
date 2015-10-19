package com.linkedin.photon.ml.supervised.classification

import breeze.linalg.Vector
import breeze.numerics.sigmoid
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

/**
 * Class for the classification model trained using Logistic Regression
 * @param coefficients Model coefficients estimated for every feature
 * @param intercept Intercept (Optional)
 * @author xazhang
 */
class LogisticRegressionModel(override val coefficients: Vector[Double], override val intercept: Option[Double])
  extends GeneralizedLinearModel(coefficients, intercept) with BinaryClassifier with Serializable {

  /**
   * Predict values for a single data point using the model trained.
   * @param features vector representing feature of a single data point's features
   * @param offset offset of the data point
   * @param threshold threshold that separates positive predictions from negative predictions. An example with prediction
   *                  score greater than or equal to this threshold is identified as positive, and negative otherwise. The default is 0.5.
   * @return predicted category from the trained model
   */
  override def predictWithOffset(features: Vector[Double], offset: Double, threshold: Double = 0.5): Double = {
    predict(coefficients, intercept, features, offset, threshold)
  }

  /**
   * Predict values for the given data points of the form RDD[(feature, offset)] with offset information using the model trained.
   * @param featuresWithOffsets data points of the form RDD[(feature, offset)]
   * @param threshold threshold that separates positive predictions from negative predictions. An example with prediction
   *                  score greater than or equal to this threshold is identified as positive, and negative otherwise. The default is 0.5.
   * @return an RDD[Double] where each entry contains the corresponding prediction
   */
  override def predictAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)], threshold: Double = 0.5): RDD[Double] = {
    val broadcastedModel = featuresWithOffsets.context.broadcast(this)
    featuresWithOffsets.map { case (features, offset) =>
      val coefficients = broadcastedModel.value.coefficients
      val intercept = broadcastedModel.value.intercept
      predict(coefficients, intercept, features, offset, threshold)
    }
  }

  private def predict(coefficients: Vector[Double], intercept: Option[Double], features: Vector[Double], offset: Double, threshold: Double): Double = {
    val score = computeMean(coefficients, intercept, features, offset)
    if (score < threshold) BinaryClassifier.negativeClassLabel else BinaryClassifier.positiveClassLabel
  }

  override protected def computeMean(coefficients: Vector[Double], intercept: Option[Double], features: Vector[Double], offset: Double): Double = {
    sigmoid(coefficients.dot(features) + intercept.getOrElse(0.0) + offset)
  }

  /**
   * Compute the classifier score given the input features of one data point with offset. The score is later used
   * to compute the area under ROC curve. For logistic regression, this score corresponds to the mean function value of
   * logistic regression, for SVM this score corresponds to the dot product between features and the estimated feature coefficients
   * @param features A data point's features
   * @param offset The offset of the data point
   * @return Computed classifier score
   */
  override def computeScoreWithOffset(features: Vector[Double], offset: Double): Double = computeMeanFunctionWithOffset(features, offset)

  /**
   * Compute the classifier scores given a RDD of input data points' features with offsets. The scores are later used
   * to compute the area under ROC curve. For logistic regression, this score corresponds to the mean function value of
   * logistic regression, for SVM this score corresponds to the dot product between features and the estimated feature coefficients
   * @param featuresWithOffsets A RDD of input data points' features with offsets (RDD[(features, offsets)])
   * @return The computed classifier scores
   */
  def computeScoresWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double] = computeMeanFunctionsWithOffsets(featuresWithOffsets)
}