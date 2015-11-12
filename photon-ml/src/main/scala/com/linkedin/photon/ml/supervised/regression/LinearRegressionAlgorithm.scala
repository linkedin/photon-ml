/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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
import com.linkedin.photon.ml.data.{ObjectProvider, LabeledPoint}
import com.linkedin.photon.ml.function.{SquaredLossFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.{LBFGS, RegularizationContext}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearAlgorithm
import com.linkedin.photon.ml.util.DataValidators
import org.apache.spark.rdd.RDD

/**
 * Train a regression model using L1/L2/Elastic net-regularized linear regression.
 *
 * @author xazhang
 * @author dpeng
 */
class LinearRegressionAlgorithm extends GeneralizedLinearAlgorithm[LinearRegressionModel, TwiceDiffFunction[LabeledPoint]] {
  override protected val validators: Seq[RDD[LabeledPoint] => Boolean] = List(DataValidators.linearRegressionValidator)

  /**
   *  Objective function = loss function + l2weight * regularization
   *  Only the L2 regularization part is implemented in the objective function. L1 part is implemented through the optimizer. See [[LBFGS]].
   */
  override protected def createObjectiveFunction(normalizationContext: ObjectProvider[NormalizationContext],
                                                 regularizationContext: RegularizationContext,
                                                 regularizationWeight: Double): TwiceDiffFunction[LabeledPoint] = {
    TwiceDiffFunction.withRegularization(new SquaredLossFunction(normalizationContext), regularizationContext, regularizationWeight)
  }

  override protected def createModel(coefficients: Vector[Double], intercept: Option[Double]) = {
    new LinearRegressionModel(coefficients, intercept)
  }
}
