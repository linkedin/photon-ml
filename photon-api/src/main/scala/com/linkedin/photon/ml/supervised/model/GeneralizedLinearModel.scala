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
package com.linkedin.photon.ml.supervised.model

import breeze.linalg.Vector
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector => SparkVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.constants.DataConst
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.util.Summarizable

/**
 * GeneralizedLinearModel (GLM) represents a model trained using GeneralizedLinearAlgorithm.
 *
 * Reference: [[http://en.wikipedia.org/wiki/Generalized_linear_model]]
 *
 * @note This class is modified based on MLLib's GeneralizedLinearModel.
 *
 * @param coefficients The generalized linear model's coefficients (or called weights in some scenarios) of the features
 */
abstract class GeneralizedLinearModel(val coefficients: Coefficients) extends Serializable with Summarizable {

  /**
   * Check the model type.
   *
   * @return The model type
   */
  def modelType: TaskType

  /**
   * Compute the mean response of the model.
   *
   * @note "mean" = linkFunction(score) (after link function in the case of logistic regression: see below)
   *
   * @param features The input data point's features
   * @param offset The input data point's offset
   * @return The mean for the passed features
   */
  protected[ml] def computeMean(features: Vector[Double], offset: Double): Double

  /**
   * Compute the value of the mean function of the generalized linear model given one data point using the estimated
   * coefficients.
   *
   * @param features Vector representing a single data point's features
   * @param offset Offset of the data point
   * @return Computed mean function value
   */
  def computeMeanFunctionWithOffset(features: Vector[Double], offset: Double): Double =
    computeMean(features, offset)

  /**
   * Create a new model of the same type with updated coefficients.
   *
   * @param updateCoefficients The new coefficients
   * @return A new generalized linear model with the passed coefficients
   */
  def updateCoefficients(updateCoefficients: Coefficients): GeneralizedLinearModel

  /**
   * Validate coefficients and offset. Child classes should add additional checks.
   */
  def validateCoefficients(): Unit = {
    val msg : StringBuilder = new StringBuilder()
    var valid : Boolean = true

    coefficients.means.foreachPair( (idx, value) => {
      if (!java.lang.Double.isFinite(value)) {
        valid = false
        msg.append("Index [" + idx + "] has value [" + value + "]\n")
      }
    })

    if (!valid) {
      throw new IllegalStateException("Detected invalid coefficients / offset: " + msg.toString())
    }
  }

  /**
   * Use String interpolation over format. It's a bit more concise and is checked at compile time (e.g. forgetting an
   * argument would be a compile error).
   *
   * @return
   */
  override def toString: String = s"coefficients: ${coefficients.means}"

  /**
   * Compares two [[GeneralizedLinearModel]] objects.
   *
   * @param other Some other object
   * @return True if the both models conform to the equality contract and have the same model coefficients, false
   *         otherwise
   */
  override def equals(other: Any): Boolean = other match {
    case that: GeneralizedLinearModel => this.coefficients == that.coefficients
    case _ => false
  }

  /**
   * Returns a hash code value for the object.
   *
   * @return An [[Int]] hash code
   */
  override def hashCode: Int = coefficients.hashCode()
}

object GeneralizedLinearModel {

  // Schema for [[DataFrame]]
  def schema: StructType = StructType(Array(
    StructField(DataConst.MODEL_ID, StringType, false),
    StructField(DataConst.MODEL_TYPE, StringType, false),
    StructField(DataConst.COEFFICIENTS, VectorType , false),
    StructField(DataConst.VARIANCES, VectorType, true)
  ))

  /**
   * Compute the value of the mean functions of the generalized linear model given a RDD of data points using the
   * estimated coefficients and intercept.
   *
   * @param features RDD representing data points' features
   * @return Computed mean function value
   */
  def computeMeanFunctions(model: GeneralizedLinearModel, features: RDD[Vector[Double]]): RDD[Double] =
    computeMeanFunctionsWithOffsets(model, features.map(feature => (feature, 0.0)))

  /**
   * Compute the value of the mean functions of a generalized linear model given a RDD of data points.
   *
   * @param model Generalized linear model to use
   * @param featuresWithOffsets Data points of the form RDD[(feature, offset)]
   * @return Computed mean function values
   */
  def computeMeanFunctionsWithOffsets(
      model: GeneralizedLinearModel,
      featuresWithOffsets: RDD[(Vector[Double], Double)]): RDD[Double] = {

    val broadcastModel = featuresWithOffsets.context.broadcast(model)
    val result = featuresWithOffsets.map { case (features, offset) =>
      broadcastModel.value.computeMeanFunctionWithOffset(features, offset)
    }

    broadcastModel.unpersist()
    result
  }

  val MODEL_TYPE = "modelType"

  /**
   * A UDF to compute scores given a linear model and a feature vector
   *
   * @return The score which is the dot product of model coefficients and features
   */
  def scoreUdf = udf[Double, SparseVector, SparseVector](
    { (coefficients: SparkVector, features: SparkVector) =>
      require(
        coefficients.size == features.size,
        s"Coefficients.size = ${coefficients.size} and features.size = ${features.size}")

      var score = 0D

      coefficients match {
        case denseCoef: DenseVector =>
          features.foreachActive { case (index, value) =>
            score += value * denseCoef(index)
          }

        case sparseCoef: SparseVector =>
          sparseCoef.foreachActive { case (index, coefficient) =>
            score += coefficient * features(index)
          }
      }

      score
    })

}
