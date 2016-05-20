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
package com.linkedin.photon.ml.supervised.regression

import breeze.linalg.Vector
import com.linkedin.photon.ml.data.{LabeledPoint, ObjectProvider}
import com.linkedin.photon.ml.function.{SquaredLossFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearAlgorithm

/**
 * Train a regression model using L1/L2/Elastic net-regularized linear regression.
 */
class LinearRegressionAlgorithm
  extends GeneralizedLinearAlgorithm[LinearRegressionModel, TwiceDiffFunction[LabeledPoint]] {

  /**
   * Create the objective function of the generalized linear algorithm.
   *
   * objective function = loss function + l2weight * regularization
   *
   * Only the L2 regularization part is implemented in the objective function. L1 part is implemented through the
   * optimizer. See [[LBFGS]].
   *
   * @param normalizationContext The normalization context for the training
   * @param regularizationContext The type of regularization to construct the objective function
   * @param regularizationWeight The weight of the regularization term in the objective function
   */
  override protected def createObjectiveFunction(
      normalizationContext: ObjectProvider[NormalizationContext],
      regularizationContext: RegularizationContext,
      regularizationWeight: Double): TwiceDiffFunction[LabeledPoint] = {

    val basicFunction = new SquaredLossFunction(normalizationContext)
    basicFunction.treeAggregateDepth = treeAggregateDepth
    TwiceDiffFunction.withRegularization(basicFunction, regularizationContext, regularizationWeight)
  }

  /**
   * Create an Optimizer according to the config.
   *
   * @param optimizerConfig Optimizer configuration
   * @return A new Optimizer created according to the configuration
   */
  override protected def createOptimizer(optimizerConfig: OptimizerConfig)
    : Optimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]] = OptimizerFactory.twiceDiffOptimizer(optimizerConfig)

  /**
   * Create a model given the coefficients
   *
   * @param coefficients The coefficients parameter of each feature
   * @return A generalized linear model with intercept and coefficients parameters
   */
  override protected def createModel(coefficients: Vector[Double]) = {
    new LinearRegressionModel(coefficients)
  }
}
