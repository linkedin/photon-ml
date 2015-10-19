package com.linkedin.photon.ml.supervised.regression

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * Represents a regression that predicts values for the given data set / data point using the model trained
 * @author xazhang
 */
trait Regression extends Serializable {

  /**
   * Predict values for the given data set using the model trained.
   *
   * @param features RDD representing data points' features
   * @return RDD[Double] where each entry contains the corresponding prediction
   */
  def predictAll(features: RDD[Vector[Double]]): RDD[Double] = {
    predictAllWithOffsets(features.map(feature => (feature, 0.0)))
  }

  /**
   * Predict values for the given data points with offsets of the form RDD[(feature, offset)] using the model trained.
   *
   * @param featuresWithOffsets data points of the form RDD[(feature, offset)]
   * @return RDD[Double] where each entry contains the corresponding prediction
   */
  def predictAllWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double]

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features vector representing a single data point's features
   * @return Double prediction from the trained model
   */
  def predict(features: Vector[Double]): Double = predictWithOffset(features, 0.0)

  /**
   * Predict values for a single data point with offset using the model trained.
   *
   * @param features vector representing feature of a single data point's features
   * @param offset offset of the data point
   * @return Double prediction from the trained model
   */
  def predictWithOffset(features: Vector[Double], offset: Double): Double
}
