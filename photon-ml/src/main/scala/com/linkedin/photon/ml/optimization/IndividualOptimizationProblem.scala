/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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

import breeze.linalg.Vector
import org.apache.spark.broadcast.Broadcast

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}

/**
 * An optimization problem solved by a single task on one executor. Used for solving the per-entity optimization
 * problems of a random effect model.
 *
 * @param optimizer The underlying optimizer which iteratively solves the convex problem
 * @param objectiveFunction The objective function to optimize
 * @param glmConstructor The function to use for producing GLMs from trained coefficients
 * @param isComputingVariances Should coefficient variances be computed in addition to the means?
 */
protected[ml] class IndividualOptimizationProblem[Function <: IndividualObjectiveFunction] protected[optimization] (
    optimizer: Optimizer[Function],
    objectiveFunction: Function,
    glmConstructor: Coefficients => GeneralizedLinearModel,
    isComputingVariances: Boolean)
  extends GeneralizedLinearOptimizationProblem[Function](
    optimizer,
    objectiveFunction,
    glmConstructor,
    isComputingVariances)
  with Serializable {

  /**
   * Compute coefficient variances
   *
   * @param input The training data
   * @param coefficients The feature coefficients means
   * @return The feature coefficient variances
   */
  override def computeVariances(input: Iterable[LabeledPoint], coefficients: Vector[Double]): Option[Vector[Double]] = {
    (isComputingVariances, objectiveFunction) match {
      case (true, twiceDiffFunc: TwiceDiffFunction) =>
        Some(twiceDiffFunc
          .hessianDiagonal(input, coefficients)
          .map(v => 1.0 / (v + MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)))

      case _ =>
        None
    }
  }

  /**
   * Run the optimization algorithm on the input data, starting from an initial model of all-0 coefficients.
   *
   * @param input The training data
   * @return The learned GLM for the given optimization problem, data, regularization type, and regularization weight
   */
  override def run(input: Iterable[LabeledPoint]): GeneralizedLinearModel =
    run(input, initializeZeroModel(input.head.features.size))

  /**
   * Run the optimization algorithm on the input data, starting from the initial model provided.
   *
   * @param input The training data
   * @param initialModel The initial model from which to begin optimization
   * @return The learned GLM for the given optimization problem, data, regularization type, and regularization weight
   */
  override def run(input: Iterable[LabeledPoint], initialModel: GeneralizedLinearModel): GeneralizedLinearModel = {
    val normalizationContext = optimizer.getNormalizationContext
    val (optimizedCoefficients, _) = optimizer.optimize(objectiveFunction, initialModel.coefficients.means)(input)
    val optimizedVariances = computeVariances(input, optimizedCoefficients)

    modelTrackerBuilder.foreach { modelTrackerBuilder =>
      val tracker = optimizer.getStateTracker.get
      logInfo(s"History tracker information:\n $tracker")
      val modelsPerIteration = tracker.getTrackedStates.map { x =>
        val coefficients = x.coefficients
        val variances = computeVariances(input, coefficients)
        createModel(normalizationContext, coefficients, variances)
      }
      logInfo(s"Number of iterations: ${modelsPerIteration.length}")
      modelTrackerBuilder += new ModelTracker(tracker.toString, modelsPerIteration)
    }

    createModel(normalizationContext, optimizedCoefficients, optimizedVariances)
  }
}

object IndividualOptimizationProblem {
  /**
   * Factory method to create new IndividualOptimizationProblems.
   *
   * @param configuration The optimization problem configuration
   * @param objectiveFunction The objective function to optimize
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param isTrackingState Should the optimization problem record the internal optimizer states?
   * @param isComputingVariance Should coefficient variances be computed in addition to the means?
   * @return A new IndividualOptimizationProblem
   */
  def createOptimizationProblem[Function <: IndividualObjectiveFunction](
    configuration: GLMOptimizationConfiguration,
    objectiveFunction: Function,
    glmConstructor: Coefficients => GeneralizedLinearModel,
    normalizationContext: Broadcast[NormalizationContext],
    isTrackingState: Boolean,
    isComputingVariance: Boolean): IndividualOptimizationProblem[Function] = {

    val optimizerConfig = configuration.optimizerConfig
    val regularizationContext = configuration.regularizationContext
    val regularizationWeight = configuration.regularizationWeight
    // Will result in a runtime error if created Optimizer cannot be cast to an Optimizer that can handle the given
    // objective function.
    val optimizer = OptimizerFactory
      .createOptimizer(optimizerConfig, normalizationContext, regularizationContext, regularizationWeight, isTrackingState)
      .asInstanceOf[Optimizer[Function]]

    new IndividualOptimizationProblem(
      optimizer,
      objectiveFunction,
      glmConstructor,
      isComputingVariance)
  }
}
