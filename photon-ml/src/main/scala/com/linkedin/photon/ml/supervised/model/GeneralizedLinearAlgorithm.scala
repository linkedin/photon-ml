/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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

import breeze.linalg.{DenseVector, SparseVector, Vector}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.normalization._
import com.linkedin.photon.ml.optimization.RegularizationContext
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization._
import com.linkedin.photon.ml.optimization.{Optimizer, RegularizationContext}
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.util.DataValidators
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * GeneralizedLinearAlgorithm implements methods to train a Generalized Linear Model (GLM).
 * This class should be extended with a loss function and the createModel function to create a new GLM.
 * @tparam GLM The type of returned generalized linear model
 * @tparam Function The type of loss function of the generalized linear algorithm
 * @author xazhang
 * @author dpeng
 */
abstract class GeneralizedLinearAlgorithm[GLM <: GeneralizedLinearModel : ClassTag, Function <: DiffFunction[LabeledPoint]]
  extends Logging with Serializable {

  /**
   * The list of data validators to check the properties of the input data, e.g., the format of the input data
   */
  protected val validators: Seq[RDD[LabeledPoint] => Boolean] = List(DataValidators.finiteFeaturesValidator, DataValidators.finiteLabelValidator)

  /**
   * Optimization state trackers
   */
  protected val modelTrackerBuilder = new mutable.ListBuffer[ModelTracker]()

  /**
   * Whether to track the optimization state (for validating and debugging purpose). Default: True.
   */
  var isTrackingState: Boolean = true

  /**
   * The target storage level if the input data get normalized.
   */
  var targetStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  /**
   * Get the optimization state trackers for the optimization problems solved in the generalized linear algorithm
   * @return Some(stateTracker) if isTrackingState is set to true, None otherwise.
   */
  def getStateTracker: Option[List[ModelTracker]] = {
    if (isTrackingState) Some(modelTrackerBuilder.toList)
    else None
  }

  /**
   * Whether to validate the input data or not (default: true)
   */
  var validateData: Boolean = true

  /**
   * Whether to enable intercept (default: false).
   * A better practice would be adding the intercept during data preparation stage, adding the intercept here will
   * create unnecessary temp memory allocation that is proportional to the input data size.
   */
  var enableIntercept: Boolean = false

  /**
   * TODO: enable feature specific regularization / disable regularizing intercept https://jira01.corp.linkedin.com:8443/browse/OFFREL-324
   * Create the objective function of the generalized linear algorithm
   * @param normalizationContext The normalization context for the training
   * @param regularizationContext The type of regularization to construct the objective function
   * @param regularizationWeight The weight of the regularization term in the objective function
   */
  protected def createObjectiveFunction(normalizationContext: NormalizationContext, regularizationContext: RegularizationContext, regularizationWeight: Double): Function

  /**
   * Create a model given the coefficients and intercept
   * @param coefficients The coefficients parameter of each feature
   * @param intercept The intercept of the generalized linear model
   * @return A generalized linear model with intercept and coefficients parameters
   */
  protected def createModel(coefficients: Vector[Double], intercept: Option[Double]): GLM

  /**
   * Create a model given the coefficients and intercept
   * @param normalizationContext The normalization context
   * @param coefficientsWithIntercept A vector of the form [intercept, coefficients parameters]
   * @return A generalized linear model with intercept and coefficients parameters
   */
  protected def createModel(normalizationContext: NormalizationContext, coefficientsWithIntercept: Vector[Double]): GLM = {
    val updatedWeights =
      if (enableIntercept) {
        new DenseVector(coefficientsWithIntercept.toArray.tail)
      } else {
        coefficientsWithIntercept
      }
    val updatedIntercept = if (enableIntercept) Some(coefficientsWithIntercept(0)) else None
    createModel(normalizationContext.transformModelCoefficients(updatedWeights), updatedIntercept)
  }

  /**
   * Run the algorithm with the configured parameters on an input RDD of LabeledPoint entries.
   * @param input A RDD of input labeled data points in the original scale
   * @param optimizer The optimizer
   * @param regularizationContext The chosen type of regularization
   * @param regularizationWeights An array of weights for the regularization term
   * @param normalizationContext The normalization context
   * @return The learned generalized linear models of each regularization weight and iteration.
   */
  def run(input: RDD[LabeledPoint],
          optimizer: Optimizer[LabeledPoint, Function],
          regularizationContext: RegularizationContext,
          regularizationWeights: List[Double],
          normalizationContext: NormalizationContext): List[GLM] = {
    val numFeatures = input.first().features.size

    val initialWeight = Vector.zeros[Double](numFeatures)
    val initialIntercept = if (enableIntercept) Some(1.0) else None
    val initialModel = createModel(initialWeight, initialIntercept)
    val models = run(input, initialModel, optimizer, regularizationContext, regularizationWeights, normalizationContext)
    models
  }

  /**
   * Run the algorithm with the configured parameters on an input RDD of LabeledPoint entries
   * starting from the initial model provided.
   * @param input A RDD of input labeled data points in the normalized scale (if normalization is enabled)
   * @param initialModel The initial model
   * @param optimizer The optimizer container to learn the models
   * @param regularizationContext The chosen type of regularization
   * @param regularizationWeights An array of weights for the regularization term
   * @param normalizationContext The normalization context
   * @return The learned generalized linear models of each regularization weight and iteration.
   */
  protected def run(input: RDD[LabeledPoint],
                    initialModel: GeneralizedLinearModel,
                    optimizer: Optimizer[LabeledPoint, Function],
                    regularizationContext: RegularizationContext,
                    regularizationWeights: List[Double],
                    normalizationContext: NormalizationContext): List[GLM] = {

    if (input.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    /* Check the data properties before running the optimizer */
    if (validateData && !validators.forall(func => func(input))) {
      throw new IllegalArgumentException("Input validation failed.")
    }

    optimizer.isTrackingState = isTrackingState

    /**
     * Prepend a scalar to a vector
     */
    def prepend(vector: Vector[Double], scalar: Double): Vector[Double] = {
      vector match {
        case dv: DenseVector[Double] => DenseVector.vertcat(new DenseVector[Double](Array(scalar)), dv)
        case sv: SparseVector[Double] => SparseVector.vertcat(new SparseVector[Double](Array(0), Array(scalar), 1), sv)
        case v: Any => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
      }
    }
    // Prepend an extra variable consisting of all 1.0's for the intercept.
    val dataPoints = if (enableIntercept) {
      input.map { case LabeledPoint(label, features, offset, weight) => LabeledPoint(label, prepend(features, 1), offset, weight) }
    } else {
      input
    }

    /* Find the path of solutions with different regularization coefficients */
    var initialCoefficientsWithIntercept =
      if (enableIntercept) prepend(initialModel.coefficients, initialModel.intercept.getOrElse(0.0))
      else initialModel.coefficients
    val models = regularizationWeights.map { regularizationWeight =>
      val objectiveFunction = createObjectiveFunction(normalizationContext, regularizationContext, regularizationWeight)
      val (coefficientsWithIntercept, _) = optimizer.optimize(dataPoints, objectiveFunction, initialCoefficientsWithIntercept)
      initialCoefficientsWithIntercept = coefficientsWithIntercept
      logInfo(s"Training model with regularization weight $regularizationWeight finished")
      if (isTrackingState) {
        val tracker = optimizer.getStatesTracker.get
        logInfo(s"History tracker information:\n $tracker")
        val modelsPerIteration = tracker.getTrackedStates.map(x => createModel(normalizationContext, x.coefficients))
        modelTrackerBuilder += new ModelTracker(tracker, modelsPerIteration)
        logInfo(s"Models number: ${modelsPerIteration.size}")
      }
      createModel(normalizationContext, coefficientsWithIntercept)
    }
    models
  }
}
