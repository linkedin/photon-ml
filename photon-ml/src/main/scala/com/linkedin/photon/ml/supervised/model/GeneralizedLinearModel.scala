package com.linkedin.photon.ml.supervised.model

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * GeneralizedLinearModel (GLM) represents a model trained using GeneralizedLinearAlgorithm.
 * Reference: [[http://en.wikipedia.org/wiki/Generalized_linear_model]].
 * Note that this class is modified based on MLLib's GeneralizedLinearModel.
 * @param coefficients The generalized linear model's coefficients (or called weights in some scenarios) of the features
 * @param intercept The generalized linear model's intercept parameter (Optional)
 * @author xazhang
 */
abstract class GeneralizedLinearModel(val coefficients: Vector[Double], val intercept: Option[Double]) extends Serializable {

  /**
   * If the generalized linear model has intercept estimated
   */
  def hasIntercept: Boolean = intercept.isDefined

  protected def computeMean(coefficients: Vector[Double], intercept: Option[Double], features: Vector[Double], offset: Double): Double

  /**
   * Compute the value of the mean function of the generalized linear model given one data point using the estimated coefficients and intercept
   * @param features vector representing a single data point's features
   * @return Computed mean function value
   */
  def computeMeanFunction(features: Vector[Double]): Double = computeMeanFunctionWithOffset(features, 0.0)

  /**
   * Compute the value of the mean function of the generalized linear model given one data point using the estimated coefficients and intercept
   * @param features vector representing a single data point's features
   * @param offset offset of the data point
   * @return Computed mean function value
   */
  def computeMeanFunctionWithOffset(features: Vector[Double], offset: Double): Double = {
    computeMean(coefficients, intercept, features, offset)
  }

  /**
   * Compute the value of the mean functions of the generalized linear model given a RDD of data points using the estimated coefficients and intercept
   * @param features RDD representing data points' features
   * @return Computed mean function value
   */
  def computeMeanFunction(features: RDD[Vector[Double]]): RDD[Double] = {
    computeMeanFunctionsWithOffsets(features.map(feature => (feature, 0.0)))
  }

  /**
   * Compute the value of the mean functions of the generalized linear model given a RDD of data points using the estimated coefficients and intercept
   * @param featuresWithOffsets Data points of the form RDD[(feature, offset)]
   * @return Computed mean function value
   */
  def computeMeanFunctionsWithOffsets(featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double] = {
    val modelBC = featuresWithOffsets.context.broadcast(this)
    featuresWithOffsets.map {
      case (features, offset) => computeMean(modelBC.value.coefficients, modelBC.value.intercept, features, offset)
    }
  }

  /**
   * Validate coefficients and offset. Child classes should add additional checks.
   */
  def validateCoefficients():Unit = {
    val msg:StringBuilder = new StringBuilder()
    var valid:Boolean = true
    coefficients.foreachPair( (idx, value) => {
      if (!java.lang.Double.isFinite(value)) {
        valid = false
        msg.append("Index [" + idx + "] has value [" + value + "]\n")
      }
    })

    intercept match {
      case None =>
      // do nothing
      case Some(value) =>
        if (!java.lang.Double.isFinite(value)) {
          msg.append("Intercept has value [" + value + "]")
          valid = false
        }
    }

    if (!valid) {
      throw new IllegalStateException("Detected invalid coefficients / offset: " + msg.toString())
    }
  }

  /**
   * Use String interpolation over format. It's a bit more concise and is checked at compile time (e.g. forgetting an argument would be a compile error).
   * @author cfreeman
   */
  override def toString: String = {
    s"intercept: $intercept, coefficients: $coefficients"
  }
}