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

import scala.collection.mutable

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.{LabeledPoint, ObjectProvider}
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.sampler.{BinaryClassificationDownSampler, DownSampler}
import com.linkedin.photon.ml.supervised.classification.SmoothedHingeLossLinearSVMModel
import com.linkedin.photon.ml.supervised.model.ModelTracker

/**
 * Optimization problem for smoothed hinge loss linear SVM
 * @param optimizer The underlying optimizer who does the job
 * @param sampler The sampler used to down-sample the training data points
 * @param objectiveFunction The objective function upon which to optimize
 * @param regularizationContext The regularization context
 * @param regularizationWeight The regularization weight of the optimization problem
 * @param modelTrackerBuilder The builder of model tracker
 * @param treeAggregateDepth The depth used in treeAggregate
 * @param isComputingVariances Whether compute variance for the learned model coefficients
 */
case class SmoothedHingeLossLinearSVMOptimizationProblem(
    optimizer: Optimizer[LabeledPoint, DiffFunction[LabeledPoint]],
    sampler: DownSampler,
    objectiveFunction: DiffFunction[LabeledPoint],
    regularizationContext: RegularizationContext,
    regularizationWeight: Double,
    modelTrackerBuilder: Option[mutable.ListBuffer[ModelTracker]],
    treeAggregateDepth: Int,
    isComputingVariances: Boolean)
  extends GeneralizedLinearOptimizationProblem[SmoothedHingeLossLinearSVMModel, DiffFunction[LabeledPoint]](
    optimizer,
    objectiveFunction,
    sampler,
    regularizationContext,
    regularizationWeight,
    modelTrackerBuilder,
    treeAggregateDepth,
    isComputingVariances) {

  /**
   * Updates properties of the objective function. Useful in cases of data-related changes or parameter sweep.
   *
   * @param normalizationContext new normalization context
   * @param regularizationWeight new regulariation weight
   * @return a new optimization problem with updated objective
   */
  override def updateObjective(
      normalizationContext: ObjectProvider[NormalizationContext],
      regularizationWeight: Double): SmoothedHingeLossLinearSVMOptimizationProblem = {

    // TODO normalization
    val lossFunction = new SmoothedHingeLossFunction
    lossFunction.treeAggregateDepth = treeAggregateDepth

    val objectiveFunction = DiffFunction.withRegularization(
      lossFunction,
      regularizationContext,
      regularizationWeight)

    SmoothedHingeLossLinearSVMOptimizationProblem(
      optimizer,
      sampler,
      objectiveFunction,
      regularizationContext,
      regularizationWeight,
      modelTrackerBuilder,
      treeAggregateDepth,
      isComputingVariances)
  }

  /**
   * Create a default smoothed hinge SVM model with 0-valued coefficients
   *
   * @param dimension The dimensionality of the model coefficients
   * @return A model with zero coefficients
   */
  override def initializeZeroModel(dimension: Int): SmoothedHingeLossLinearSVMModel =
    SmoothedHingeLossLinearSVMOptimizationProblem.initializeZeroModel(dimension)

  /**
   * Create a model given the coefficients
   *
   * @param coefficients The coefficients parameter of each feature (and potentially including intercept)
   * @param variances The coefficient variances
   * @return A generalized linear model with coefficients parameters
   */
  override protected[optimization] def createModel(
      coefficients: Vector[Double],
      variances: Option[Vector[Double]]): SmoothedHingeLossLinearSVMModel =
    new SmoothedHingeLossLinearSVMModel(Coefficients(coefficients, variances))

  /**
   * Compute coefficient variances
   *
   * @param labeledPoints The training dataset
   * @param coefficients The model coefficients
   * @return The coefficient variances
   */
  override protected[optimization] def computeVariances(
      labeledPoints: RDD[LabeledPoint],
      coefficients: Vector[Double]): Option[Vector[Double]] = {

    logInfo("SmoothedHingeLossLinearSVMOptimizationProblem does not support coefficient variances.")
    None
  }

  /**
   * Compute coefficient variances
   *
   * @param labeledPoints The training dataset
   * @param coefficients The model coefficients
   * @return The coefficient variances
   */
  override protected[optimization] def computeVariances(
      labeledPoints: Iterable[LabeledPoint],
      coefficients: Vector[Double]): Option[Vector[Double]] = {

    logInfo("SmoothedHingeLossLinearSVMOptimizationProblem does not support coefficient variances.")
    None
  }
}

object SmoothedHingeLossLinearSVMOptimizationProblem {
  /**
   * Build a logistic regression optimization problem
   *
   * @param configuration The optimizer configuration
   * @param treeAggregateDepth The Spark tree aggregation depth
   * @param isTrackingState Should intermediate model states be tracked?
   * @return A logistic regression optimization problem instance
   */
  protected[ml] def buildOptimizationProblem(
      configuration: GLMOptimizationConfiguration,
      treeAggregateDepth: Int = 1,
      isTrackingState: Boolean = true,
      isComputingVariance: Boolean = false): SmoothedHingeLossLinearSVMOptimizationProblem = {

    val optimizerConfig = configuration.optimizerConfig
    val regularizationContext = configuration.regularizationContext
    val regularizationWeight = configuration.regularizationWeight
    val downSamplingRate = configuration.downSamplingRate

    val optimizer = OptimizerFactory.diffOptimizer(optimizerConfig)
    val sampler = new BinaryClassificationDownSampler(downSamplingRate)
    val lossFunction = new SmoothedHingeLossFunction
    lossFunction.treeAggregateDepth = treeAggregateDepth
    val objectiveFunction = DiffFunction.withRegularization(
      lossFunction,
      regularizationContext,
      regularizationWeight)

    SmoothedHingeLossLinearSVMOptimizationProblem(
      optimizer,
      sampler,
      objectiveFunction,
      regularizationContext,
      regularizationWeight,
      if (isTrackingState) { Some(new mutable.ListBuffer[ModelTracker]())} else { None },
      treeAggregateDepth,
      isComputingVariance)
  }

  def initializeZeroModel(dimension: Int): SmoothedHingeLossLinearSVMModel =
    new SmoothedHingeLossLinearSVMModel(Coefficients.initializeZeroCoefficients(dimension))
}
