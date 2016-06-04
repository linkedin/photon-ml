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

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.sampler.{BinaryClassificationDownSampler, DownSampler}
import com.linkedin.photon.ml.supervised.classification.SmoothedHingeLossLinearSVMModel
import com.linkedin.photon.ml.supervised.model.ModelTracker

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

  override def updateRegularizationWeight(updatedRegularizationWeight: Double)
    : SmoothedHingeLossLinearSVMOptimizationProblem = {

    val lossFunction = new LogisticLossFunction
    lossFunction.treeAggregateDepth = treeAggregateDepth
    val objectiveFunction = TwiceDiffFunction.withRegularization(
      lossFunction,
      regularizationContext,
      updatedRegularizationWeight)

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

  override def initializeZeroModel(dimension: Int): SmoothedHingeLossLinearSVMModel =
    new SmoothedHingeLossLinearSVMModel(Coefficients.initializeZeroCoefficients(dimension))

  override protected def createModel(coefficients: Vector[Double], variances: Option[Vector[Double]])
  : SmoothedHingeLossLinearSVMModel = new SmoothedHingeLossLinearSVMModel(Coefficients(coefficients, variances))

  override protected def computeVariances(labeledPoints: RDD[LabeledPoint], coefficients: Vector[Double])
  : Option[Vector[Double]] = {

    logInfo("SmoothedHingeLossLinearSVMOptimizationProblem does not support coefficient variances.")
    None
  }

  override protected def computeVariances(labeledPoints: Iterable[LabeledPoint], coefficients: Vector[Double])
  : Option[Vector[Double]] = {

    logInfo("SmoothedHingeLossLinearSVMOptimizationProblem does not support coefficient variances.")
    None
  }
}

object SmoothedHingeLossLinearSVMOptimizationProblem {
  val COMPUTING_VARIANCE = false

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
      isTrackingState: Boolean = true): SmoothedHingeLossLinearSVMOptimizationProblem = {

    val optimizerConfig = configuration.optimizerConfig
    val regularizationContext = configuration.regularizationContext
    val regularizationWeight = configuration.regularizationWeight
    val downSamplingRate = configuration.downSamplingRate

    val optimizer = OptimizerFactory.diffOptimizer(optimizerConfig)
    val sampler = new BinaryClassificationDownSampler(downSamplingRate)
    val lossFunction = new LogisticLossFunction
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
      COMPUTING_VARIANCE)
  }
}
