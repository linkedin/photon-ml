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

import breeze.linalg.{Vector, cholesky, diag}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.BroadcastWrapper
import com.linkedin.photon.ml.util.Linalg.choleskyInverse

/**
 * An optimization problem solved by a single task on one executor. Used for solving the per-entity optimization
 * problems of a random effect model.
 *
 * @tparam Objective The objective function to optimize, using a single node
 * @param optimizer The underlying optimizer which iteratively solves the convex problem
 * @param objectiveFunction The objective function to optimize
 * @param glmConstructor The function to use for producing GLMs from trained coefficients
 * @param varianceComputationType If an how to compute coefficient variances
 */
protected[ml] class SingleNodeOptimizationProblem[Objective <: SingleNodeObjectiveFunction] protected[optimization] (
    optimizer: Optimizer[Objective],
    objectiveFunction: Objective,
    glmConstructor: Coefficients => GeneralizedLinearModel,
    varianceComputationType: VarianceComputationType)
  extends GeneralizedLinearOptimizationProblem[Objective](
    optimizer,
    objectiveFunction,
    glmConstructor,
    varianceComputationType)
  with Serializable {

  /**
   * Compute coefficient variances (if enabled).
   *
   * @param input The training data
   * @param coefficients The feature coefficients means
   * @return An optional feature coefficient variances vector
   */
  override def computeVariances(input: Iterable[LabeledPoint], coefficients: Vector[Double]): Option[Vector[Double]] =
    (objectiveFunction, varianceComputationType) match {
      case (twiceDiffFunc: TwiceDiffFunction, VarianceComputationType.SIMPLE) =>
        Some(twiceDiffFunc
          .hessianDiagonal(input, coefficients)
          .map(v => 1.0 / math.max(v, MathConst.EPSILON)))

      case (twiceDiffFunc: TwiceDiffFunction, VarianceComputationType.FULL) =>
        val hessianMatrix = twiceDiffFunc.hessianMatrix(input, coefficients)
        val invHessianMatrix = choleskyInverse(cholesky(hessianMatrix))

        Some(diag(invHessianMatrix))

      case _ =>
        None
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

    createModel(normalizationContext, optimizedCoefficients, optimizedVariances)
  }
}

object SingleNodeOptimizationProblem {

  /**
   * Factory method to create new SingleNodeOptimizationProblems.
   *
   * @param configuration The optimization problem configuration
   * @param objectiveFunction The objective function to optimize
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param varianceComputationType Whether to compute coefficient variances, and if so how
   * @return A new [[SingleNodeOptimizationProblem]]
   */
  def apply[Function <: SingleNodeObjectiveFunction](
      configuration: GLMOptimizationConfiguration,
      objectiveFunction: Function,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: BroadcastWrapper[NormalizationContext],
      varianceComputationType: VarianceComputationType): SingleNodeOptimizationProblem[Function] = {

    val optimizerConfig = configuration.optimizerConfig
    val regularizationContext = configuration.regularizationContext
    val regularizationWeight = configuration.regularizationWeight
    // Will result in a runtime error if created Optimizer cannot be cast to an Optimizer that can handle the given
    // objective function.
    val optimizer = OptimizerFactory
      .build(optimizerConfig, normalizationContext, regularizationContext, regularizationWeight)
      .asInstanceOf[Optimizer[Function]]

    new SingleNodeOptimizationProblem(
      optimizer,
      objectiveFunction,
      glmConstructor,
      varianceComputationType)
  }
}
