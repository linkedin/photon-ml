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

import breeze.linalg.Vector
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.{MathConst, StorageLevel}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.{DistributedObjectiveFunction, L2Regularization, TwiceDiffFunction}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.sampler.DownSampler
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}

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
 * @param isComputingVariances Should coefficient variances be computed in addition to the means?
 */
protected[ml] class DistributedOptimizationProblem[Objective <: DistributedObjectiveFunction] protected[optimization] (
    optimizer: Optimizer[Objective],
    objectiveFunction: Objective,
    samplerOption: Option[DownSampler],
    glmConstructor: Coefficients => GeneralizedLinearModel,
    regularizationContext: RegularizationContext,
    isComputingVariances: Boolean)
  extends GeneralizedLinearOptimizationProblem[Objective](
    optimizer,
    objectiveFunction,
    glmConstructor,
    isComputingVariances) {

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
   * Compute coefficient variances
   *
   * @param input The training data
   * @param coefficients The feature coefficients means
   * @return The feature coefficient variances
   */
  override def computeVariances(input: RDD[LabeledPoint], coefficients: Vector[Double]): Option[Vector[Double]] = {
    (isComputingVariances, objectiveFunction) match {
      case (true, twiceDiffFunc: TwiceDiffFunction) =>
        val broadcastCoefficients = input.sparkContext.broadcast(coefficients)
        val result = Some(twiceDiffFunc
          .hessianDiagonal(input, broadcastCoefficients)
          .map(v => 1.0 / (v + MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)))

        broadcastCoefficients.unpersist()
        result

      case _ =>
        None
    }
  }

  /**
   * Run the algorithm with the configured parameters, starting from an initial model of all-0 coefficients.
   *
   * @param input The training data
   * @return The learned generalized linear models of each regularization weight and iteration.
   */
  override def run(input: RDD[LabeledPoint]): GeneralizedLinearModel =
    run(input, initializeZeroModel(input.first.features.size))

  /**
   * Run the algorithm with the configured parameters, starting from the initial model provided, and down-sample the
   * input training data first.
   *
   * @param input The training data
   * @param initialModel The initial model from which to begin optimization
   * @return The learned generalized linear models of each regularization weight and iteration.
   */
  def runWithSampling(input: RDD[(Long, LabeledPoint)], initialModel: GeneralizedLinearModel): GeneralizedLinearModel = {
    val data = (samplerOption match {
        case Some(sampler) => sampler.downSample(input).values
        case None => input.values
      })
      .setName("In memory fixed effect training data set")
      .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val result = run(data, initialModel)

    data.unpersist()

    result
  }

  /**
   * Run the algorithm with the configured parameters, starting from the initial model provided.
   *
   * @param input The training data
   * @param initialModel The initial model from which to begin optimization
   * @return The learned generalized linear models of each regularization weight and iteration.
   */
  override def run(input: RDD[LabeledPoint], initialModel: GeneralizedLinearModel): GeneralizedLinearModel = {
    val normalizationContext = optimizer.getNormalizationContext
    val (optimizedCoefficients, _) = optimizer.optimize(objectiveFunction, initialModel.coefficients.means)(input)
    val optimizedVariances = computeVariances(input, optimizedCoefficients)

    modelTrackerBuilder.foreach { modelTrackerBuilder =>
      val tracker = optimizer.getStateTracker.get
      logger.info(s"History tracker information:\n $tracker")
      val modelsPerIteration = tracker.getTrackedStates.map { x =>
        val coefficients = x.coefficients
        val variances = computeVariances(input, coefficients)
        createModel(normalizationContext, coefficients, variances)
      }
      logger.info(s"Number of iterations: ${modelsPerIteration.length}")
      modelTrackerBuilder += new ModelTracker(tracker, modelsPerIteration)
    }

    createModel(normalizationContext, optimizedCoefficients, optimizedVariances)
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
   * @param isTrackingState Should the optimization problem record the internal optimizer states?
   * @param isComputingVariance Should coefficient variances be computed in addition to the means?
   * @return A new DistributedOptimizationProblem
   */
  def create[Function <: DistributedObjectiveFunction](
    configuration: GLMOptimizationConfiguration,
    objectiveFunction: Function,
    samplerOption: Option[DownSampler],
    glmConstructor: Coefficients => GeneralizedLinearModel,
    normalizationContext: Broadcast[NormalizationContext],
    isTrackingState: Boolean,
    isComputingVariance: Boolean): DistributedOptimizationProblem[Function] = {

    val optimizerConfig = configuration.optimizerConfig
    val regularizationContext = configuration.regularizationContext
    val regularizationWeight = configuration.regularizationWeight
    // Will result in a runtime error if created Optimizer cannot be cast to an Optimizer that can handle the given
    // objective function.
    val optimizer = OptimizerFactory
      .build(optimizerConfig, normalizationContext, regularizationContext, regularizationWeight, isTrackingState)
      .asInstanceOf[Optimizer[Function]]

    new DistributedOptimizationProblem(
      optimizer,
      objectiveFunction,
      samplerOption,
      glmConstructor,
      regularizationContext,
      isComputingVariance)
  }
}
