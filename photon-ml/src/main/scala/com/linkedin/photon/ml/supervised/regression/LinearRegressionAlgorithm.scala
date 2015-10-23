package com.linkedin.photon.ml.supervised.regression

import breeze.linalg.Vector
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.{SquaredLossFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.{LBFGS, RegularizationContext}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearAlgorithm

/**
 * Train a regression model using L1/L2/Elastic net-regularized linear regression.
 *
 * @author xazhang
 * @author dpeng
 */
class LinearRegressionAlgorithm extends GeneralizedLinearAlgorithm[LinearRegressionModel, TwiceDiffFunction[LabeledPoint]] {

  /**
   *  Objective function = loss function + l2weight * regularization
   *  Only the L2 regularization part is implemented in the objective function. L1 part is implemented through the optimizer. See [[LBFGS]].
   */
  override protected def createObjectiveFunction(normalizationContext: NormalizationContext, regularizationContext: RegularizationContext, regularizationWeight: Double): TwiceDiffFunction[LabeledPoint] = {
    TwiceDiffFunction.withRegularization(new SquaredLossFunction(normalizationContext), regularizationContext, regularizationWeight)
  }

  override protected def createModel(coefficients: Vector[Double], intercept: Option[Double]) = {
    new LinearRegressionModel(coefficients, intercept)
  }
}
