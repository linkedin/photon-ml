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
package com.linkedin.photon.ml.optimization

import scala.collection.mutable
import scala.math.abs

import breeze.linalg.{Vector, sum}
import org.apache.spark.broadcast.Broadcast

import com.linkedin.photon.ml.function.{L2Regularization, ObjectiveFunction}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}
import com.linkedin.photon.ml.util.Logging

/**
 * An abstract base for the convex optimization problem which produce trained generalized linear models (GLMs) when
 * solved.
 *
 * @tparam Objective The objective function to optimize
 * @param optimizer The underlying optimizer which iteratively solves the convex problem
 * @param objectiveFunction The objective function to optimize
 * @param glmConstructor The function to use for producing GLMs from trained coefficients
 * @param isComputingVariances Should coefficient variances be computed in addition to the means?
 */
protected[ml] abstract class GeneralizedLinearOptimizationProblem[Objective <: ObjectiveFunction](
    optimizer: Optimizer[Objective],
    objectiveFunction: Objective,
    glmConstructor: Coefficients => GeneralizedLinearModel,
    isComputingVariances: Boolean) extends Logging {

  protected val modelTrackerBuilder: Option[mutable.ListBuffer[ModelTracker]] =
    if (optimizer.isTrackingState) Some(new mutable.ListBuffer[ModelTracker]()) else None

  /**
   * Get the optimization state trackers for the optimization problems solved
   *
   * @return Some(OptimizationStatesTracker) if optimization states were tracked, otherwise None
   */
  def getStatesTracker: Option[OptimizationStatesTracker] = optimizer.getStateTracker

  /**
   * Get models for the intermediate optimization states of the optimization problems solved
   *
   * @return Some(List[ModelTrackers]) if optimization states were tracked, otherwise None
   */
  def getModelTracker: Option[List[ModelTracker]] = modelTrackerBuilder.map(_.toList)

  /**
   * Create a default generalized linear model with 0-valued coefficients
   *
   * @param dimension The dimensionality of the model coefficients
   * @return A model with zero coefficients
   */
  def initializeZeroModel(dimension: Int): GeneralizedLinearModel =
    glmConstructor(Coefficients.initializeZeroCoefficients(dimension))

  /**
   * Create a GLM from given coefficients (potentially including intercept)
   *
   * @param coefficients The feature coefficients means
   * @param variances The feature coefficient variances
   * @return A GLM with the provided feature coefficients
   */
  protected def createModel(coefficients: Vector[Double], variances: Option[Vector[Double]]): GeneralizedLinearModel =
    glmConstructor(Coefficients(coefficients, variances))

  /**
   * Create a GLM from given normalized coefficients (potentially including intercept)
   *
   * @param normalizationContext The normalization context
   * @param coefficients The feature coefficients means
   * @param variances The feature coefficient variances
   * @return A GLM with the provided feature coefficients
   */
  protected def createModel(
      normalizationContext: Broadcast[NormalizationContext],
      coefficients: Vector[Double],
      variances: Option[Vector[Double]]): GeneralizedLinearModel =
    createModel(
      normalizationContext.value.modelToOriginalSpace(coefficients),
      variances.map(normalizationContext.value.modelToOriginalSpace))

  /**
   * Compute coefficient variances
   *
   * @param input The training data
   * @param coefficients The feature coefficients means
   * @return The feature coefficient variances
   */
  def computeVariances(input: objectiveFunction.Data, coefficients: Vector[Double]): Option[Vector[Double]]

  /**
   * Run the optimization algorithm on the input data, starting from an initial model of all-0 coefficients.
   *
   * @param input The training data
   * @return The learned GLM for the given optimization problem, data, regularization type, and regularization weight
   */
  def run(input: objectiveFunction.Data): GeneralizedLinearModel

  /**
   * Run the optimization algorithm on the input data, starting from the initial model provided.
   *
   * @param input The training data
   * @param initialModel The initial model from which to begin optimization
   * @return The learned GLM for the given optimization problem, data, regularization type, and regularization weight
   */
  def run(input: objectiveFunction.Data, initialModel: GeneralizedLinearModel): GeneralizedLinearModel

  /**
   * Compute the regularization term value
   *
   * @param model A trained GLM
   * @return The regularization term value of this optimization problem for the given GLM
   */
  def getRegularizationTermValue(model: GeneralizedLinearModel): Double = {
    import GeneralizedLinearOptimizationProblem._

    val l1RegValue = optimizer match {
      case l1Optimizer: OWLQN => getL1RegularizationTermValue(model, l1Optimizer.l1RegularizationWeight)
      case _ => 0D
    }
    val l2RegValue = objectiveFunction match {
      case l2ObjFunc: L2Regularization =>
        getL2RegularizationTermValue(model, l2ObjFunc.l2RegularizationWeight)
      case _ => 0D
    }

    l1RegValue + l2RegValue
  }
}

object GeneralizedLinearOptimizationProblem {
  /**
   * Compute the L1 regularization term value
   *
   * @param model the model
   * @param regularizationWeight the weight of the regularization value
   * @return L1 regularization term value
   */
  protected[ml] def getL1RegularizationTermValue(model: GeneralizedLinearModel, regularizationWeight: Double)
    : Double = {

    val coefficients = model.coefficients.means
    sum(coefficients.map(abs)) * regularizationWeight
  }

  /**
   * Compute the L2 regularization term value
   *
   * @param model the model
   * @param regularizationWeight the weight of the regularization value
   * @return L2 regularization term value
   */
  protected[ml] def getL2RegularizationTermValue(model: GeneralizedLinearModel, regularizationWeight: Double)
    : Double = {

    val coefficients = model.coefficients.means
    coefficients.dot(coefficients) * regularizationWeight / 2
  }
}
