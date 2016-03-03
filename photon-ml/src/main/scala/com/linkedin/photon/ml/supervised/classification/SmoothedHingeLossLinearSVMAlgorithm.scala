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
package com.linkedin.photon.ml.supervised.classification

import breeze.linalg.Vector
import com.linkedin.photon.ml.data.{LabeledPoint, ObjectProvider}
import com.linkedin.photon.ml.function.{DiffFunction, SmoothedHingeLossFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.{Optimizer, OptimizerConfig, OptimizerFactory, RegularizationContext}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearAlgorithm

/**
 * Approximate linear SVM via soft hinge loss
 */
class SmoothedHingeLossLinearSVMAlgorithm
  extends GeneralizedLinearAlgorithm[SmoothedHingeLossLinearSVMModel, DiffFunction[LabeledPoint]] {

  /**
   * TODO: enable feature specific regularization / disable regularizing intercept
   *   https://jira01.corp.linkedin.com:8443/browse/OFFREL-324
   * Create the objective function of the generalized linear algorithm
   * @param normalizationContext The normalization context for the training
   * @param regularizationContext The type of regularization to construct the objective function
   * @param regularizationWeight The weight of the regularization term in the objective function
   */
  override protected def createObjectiveFunction(
      normalizationContext: ObjectProvider[NormalizationContext],
      regularizationContext: RegularizationContext,
      regularizationWeight: Double): DiffFunction[LabeledPoint] = {
    // Ignore normalization for now -- not clear what refactoring is necessary / appropriate to make this fit within
    // Degao's normalization framework.
    val basicFunction = new SmoothedHingeLossFunction()
    basicFunction.treeAggregateDepth = treeAggregateDepth
    DiffFunction.withRegularization(basicFunction, regularizationContext, regularizationWeight)
  }

  /**
   * Create an Optimizer according to the config.
   *
   * @param optimizerConfig Optimizer configuration
   * @return A new Optimizer created according to the configuration
   */
  override protected def createOptimizer(
      optimizerConfig: OptimizerConfig): Optimizer[LabeledPoint, DiffFunction[LabeledPoint]] = {
    OptimizerFactory.diffOptimizer(optimizerConfig)
  }

  /**
   * Create a model given the coefficients and intercept
   * @param coefficients The coefficients parameter of each feature
   * @param intercept The intercept of the generalized linear model
   * @return A generalized linear model with intercept and coefficients parameters
   */
  override protected def createModel(
      coefficients: Vector[Double],
      intercept: Option[Double]): SmoothedHingeLossLinearSVMModel = {
    new SmoothedHingeLossLinearSVMModel(coefficients, intercept)
  }
}
