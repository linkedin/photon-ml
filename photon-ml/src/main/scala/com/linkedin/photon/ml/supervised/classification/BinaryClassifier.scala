package com.linkedin.photon.ml.supervised.classification

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * Represents a binary classifier, with 1 representing the positive label and 0 representing the negative label.
 * @author xazhang
 */
trait BinaryClassifier extends Serializable {

  /**
   * Predict values for a single data point using the model trained.
   * @param features vector representing a single data point's features
   * @param threshold threshold that separates positive predictions from negative predictions. An example with prediction
   *                  score greater than or equal to this threshold is identified as positive, and negative otherwise.
   * @return predicted category from the trained model
   */
  def predict(features: Vector[Double], threshold: Double): Double = predictWithOffset(features, 0.0, threshold)

  /**
   * Predict values for a single data point with offset using the model trained.
   * @param features vector representing feature of a single data point's features
   * @param offset offset of the data point
   * @param threshold threshold that separates positive predictions from negative predictions. An example with prediction
   *                  score greater than or equal to this threshold is identified as positive, and negative otherwise.
   * @return predicted category from the trained model
   */
  def predictWithOffset(features: Vector[Double], offset: Double, threshold: Double): Double

  /**
   * Predict values for the given data points of the form RDD[feature] using the model trained.
   * @param features RDD representing data points' features
   * @param threshold threshold that separates positive predictions from negative predictions. An example with prediction
   *                  score greater than or equal to this threshold is identified as positive, and negative otherwise.
   * @return an RDD[Double] where each entry contains the corresponding prediction
   */
  def predictAll(features: RDD[Vector[Double]], threshold: Double): RDD[Double] = {
    predictAllWithOffsets(features.map(feature => (feature, 0.0)), threshold)
  }

  /**
   * Predict values for the given data points with offsets of the form RDD[(feature, offset)] using the model trained.
   * @param featuresWithOffsets data points of the form RDD[(feature, offset)]
   * @param threshold threshold that separates positive predictions from negative predictions. An example with prediction
   *                  score greater than or equal to this threshold is identified as positive, and negative otherwise.
   * @return an RDD[Double] where each entry contains the corresponding prediction
   */
  def predictAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)], threshold: Double): RDD[Double]

  /**
   * Compute the classifier score given the input features of one data point. The score is later used
   * to compute the area under ROC curve. For logistic regression, this score corresponds to the mean function value of
   * logistic regression, for SVM this score corresponds to the dot product between features and the estimated feature coefficients
   * @param features A data point's features
   * @return The computed classifier score
   */
  def computeScore(features: Vector[Double]): Double = computeScoreWithOffset(features, 0.0)

  /**
   * Compute the classifier score given the input features of one data point with offset. The score is later used
   * to compute the area under ROC curve. For logistic regression, this score corresponds to the mean function value of
   * logistic regression, for SVM this score corresponds to the dot product between features and the estimated feature coefficients
   * @param features A data point's features
   * @param offset The offset of the data point
   * @return Computed classifier score
   */
  def computeScoreWithOffset(features: Vector[Double], offset: Double): Double

  /**
   * Compute the classifier scores given a RDD of input data points' features. The scores are later used
   * to compute the area under ROC curve. For logistic regression, this score corresponds to the mean function value of
   * logistic regression, for SVM this score corresponds to the dot product between features and the estimated feature coefficients
   * @param features A RDD of input data points' features
   * @return The computed classifier scores
   */
  def computeScores(features: RDD[Vector[Double]]): RDD[Double] = computeScoresWithOffsets(features.map(feature => (feature, 0.0)))

  /**
   * Compute the classifier scores given a RDD of input data points' features with offsets. The scores are later used
   * to compute the area under ROC curve. For logistic regression, this score corresponds to the mean function value of
   * logistic regression, for SVM this score corresponds to the dot product between features and the estimated feature coefficients
   * @param featuresWithOffsets A RDD of input data points' features with offsets (RDD[(features, offsets)])
   * @return The computed classifier scores
   */
  def computeScoresWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double]
}

object BinaryClassifier {
  val positiveClassLabel = 1.0
  val negativeClassLabel = 0.0
}