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
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.{DistributedObjectiveFunction, L2Regularization, TwiceDiffFunction}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.sampling.DownSampler
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.{BroadcastWrapper, VectorUtils}
import com.linkedin.photon.ml.util.Linalg.choleskyInverse

/**
 * An optimization problem solved by multiple tasks on one or more executors. Used for solving the global optimization
 * problems of a fixed effect model.
 *
 * @tparam Objective The objective function to optimize, using one or more nodes (executors)
 * @param optimizer The underlying optimizer which iteratively solves the convex problem
 * @param objectiveFunction The objective function to optimize
 * @param samplerOption (Optional) A sampler to use for down-sampling the training data prior to optimization
 * @param glmConstructor The function to use for producing GLMs from trained coefficients
 * @param regularizationContext The regularization context
 * @param varianceComputation If an how to compute coefficient variances
 */
protected[ml] class DistributedOptimizationProblem[Objective <: DistributedObjectiveFunction] protected[optimization](
    optimizer: Optimizer[Objective],
    objectiveFunction: Objective,
    samplerOption: Option[DownSampler],
    glmConstructor: Coefficients => GeneralizedLinearModel,
    regularizationContext: RegularizationContext,
    varianceComputation: VarianceComputationType)
  extends GeneralizedLinearOptimizationProblem[Objective](
    optimizer,
    objectiveFunction,
    glmConstructor,
    varianceComputation) {

  /**
   * Update the regularization weight for the optimization problem
   *
   * @param regularizationWeight The new regularization weight
   */
  def updateRegularizationWeight(regularizationWeight: Double): Unit = {

    optimizer match {
      case owlqn: OWLQN =>
        owlqn.l1RegularizationWeight = regularizationContext.getL1RegularizationWeight(regularizationWeight)
      case _ =>
    }

    objectiveFunction match {
      case l2RegFunc: DistributedObjectiveFunction with L2Regularization =>
        l2RegFunc.l2RegularizationWeight = regularizationContext.getL2RegularizationWeight(regularizationWeight)
      case _ =>
    }
  }

  /**
   * Compute coefficient variances (if enabled).
   *
   * @param input The training data
   * @param coefficients The feature coefficients means
   * @return An optional feature coefficient variances vector
   */
  override def computeVariances(input: RDD[LabeledPoint], coefficients: Vector[Double]): Option[Vector[Double]] = {

    val broadcastCoefficients = input.sparkContext.broadcast(coefficients)

    val result = (objectiveFunction, varianceComputation) match {
      case (twiceDiffFunc: TwiceDiffFunction, VarianceComputationType.SIMPLE) =>
        Some(VectorUtils.invertVector(twiceDiffFunc.hessianDiagonal(input, broadcastCoefficients)))

      case (twiceDiffFunc: TwiceDiffFunction, VarianceComputationType.FULL) =>
        val hessianMatrix = twiceDiffFunc.hessianMatrix(input, broadcastCoefficients)
        val invHessianMatrix = choleskyInverse(cholesky(hessianMatrix))

        Some(diag(invHessianMatrix))

      case _ =>
        None
    }

    broadcastCoefficients.unpersist()

    result
  }

  /**
   * Run the algorithm with the configured parameters, starting from an initial model of all-0 coefficients
   * (cold start in iterations over the regularization weights for hyperparameter tuning).
   *
   * @param input The training data
   * @return The learned [[GeneralizedLinearModel]]
   */
  override def run(input: RDD[LabeledPoint]): GeneralizedLinearModel =
    run(input, initializeZeroModel(input.first.features.size))

  /**
   * Run the algorithm with the configured parameters, starting from the initial model provided
   * (warm start in iterations over the regularization weights for hyperparameter tuning).
   *
   * @param input The training data
   * @param initialModel The initial model from which to begin optimization
   * @return The learned [[GeneralizedLinearModel]]
   */
  override def run(input: RDD[LabeledPoint], initialModel: GeneralizedLinearModel): GeneralizedLinearModel = {

    val normalizationContext = optimizer.getNormalizationContext
    val (optimizedCoefficients, _) = optimizer.optimize(objectiveFunction, initialModel.coefficients.means)(input)
    val optimizedVariances = computeVariances(input, optimizedCoefficients)

    createModel(normalizationContext, optimizedCoefficients, optimizedVariances)
  }

  /**
   * Run the algorithm with the configured parameters, starting from an initial model of all-0 coefficients, and
   * down-sample the input training data first.
   *
   * @param input The training data
   * @return The learned [[GeneralizedLinearModel]]
   */
  def runWithSampling(input: RDD[(UniqueSampleId, LabeledPoint)]): GeneralizedLinearModel =
    runWithSampling(input, initializeZeroModel(input.first._2.features.size))

  /**
   * Run the algorithm with the configured parameters, starting from the initial model provided, and down-sample the
   * input training data first.
   *
   * @param input The training data
   * @param initialModel The initial model from which to begin optimization
   * @return The learned [[GeneralizedLinearModel]]
   */
  def runWithSampling(
      input: RDD[(UniqueSampleId, LabeledPoint)],
      initialModel: GeneralizedLinearModel): GeneralizedLinearModel = {

    val data = (samplerOption match {
        case Some(sampler) => sampler.downSample(input).values
        case None => input.values
      })
      .setName("In memory fixed effect training dataset")
      .persist(StorageLevel.MEMORY_AND_DISK)
    val result = run(data, initialModel)

    data.unpersist()

    result
  }
}

object DistributedOptimizationProblem {

  /**
   * Factory method to create new DistributedOptimizationProblems.
   *
   * @param configuration The optimization problem configuration
   * @param objectiveFunction The objective function to optimize
   * @param samplerOption (Optional) A sampler to use for down-sampling the training data prior to optimization
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param varianceComputation If and how coefficient variances should be computed
   * @return A new [[DistributedOptimizationProblem]]
   */
  def apply[Function <: DistributedObjectiveFunction](
      configuration: GLMOptimizationConfiguration,
      objectiveFunction: Function,
      samplerOption: Option[DownSampler],
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: BroadcastWrapper[NormalizationContext],
      varianceComputation: VarianceComputationType): DistributedOptimizationProblem[Function] = {

    val optimizerConfig = configuration.optimizerConfig
    val regularizationContext = configuration.regularizationContext
    val regularizationWeight = configuration.regularizationWeight
    // Will result in a runtime error if created Optimizer cannot be cast to an Optimizer that can handle the given
    // objective function.
    val optimizer = OptimizerFactory
      .build(optimizerConfig, normalizationContext, regularizationContext, regularizationWeight)
      .asInstanceOf[Optimizer[Function]]

    new DistributedOptimizationProblem(
      optimizer,
      objectiveFunction,
      samplerOption,
      glmConstructor,
      regularizationContext,
      varianceComputation)
  }
}
