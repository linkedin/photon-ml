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
import com.linkedin.photon.ml.data.{DataValidators, ObjectProvider, LabeledPoint}
import com.linkedin.photon.ml.function.{PoissonLossFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.{Optimizer, OptimizerConfig, OptimizerFactory, RegularizationContext}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearAlgorithm
import org.apache.spark.rdd.RDD


/**
 * Train a regression model using L2-regularized poisson regression.
 * @author asaha
 * @author dpeng
 */
class PoissonRegressionAlgorithm
  extends GeneralizedLinearAlgorithm[PoissonRegressionModel, TwiceDiffFunction[LabeledPoint]] {

  /* Objective function = loss function + l2weight * regularization */
  override protected def createObjectiveFunction(
      normalizationContext: ObjectProvider[NormalizationContext],
      regularizationContext: RegularizationContext,
      regularizationWeight: Double): TwiceDiffFunction[LabeledPoint] = {
    val basicFunction = new PoissonLossFunction(normalizationContext)
    basicFunction.treeAggregateDepth = treeAggregateDepth
    TwiceDiffFunction.withRegularization(basicFunction, regularizationContext, regularizationWeight)
  }

  /**
   * Create an Optimizer according to the config.
   *
   * @param optimizerConfig Optimizer configuration
   * @return A new Optimizer created according to the configuration
   */
  override protected def createOptimizer(
      optimizerConfig: OptimizerConfig): Optimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]] = {
    OptimizerFactory.twiceDiffOptimizer(optimizerConfig)
  }

  /**
   * Create a poisson regression model given the estimated coefficients and intercept
   */
  override protected def createModel(coefficients: Vector[Double], intercept: Option[Double]) = {
    new PoissonRegressionModel(coefficients, intercept)
  }
}
