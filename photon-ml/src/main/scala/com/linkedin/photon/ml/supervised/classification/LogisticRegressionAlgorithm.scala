package com.linkedin.photon.ml.supervised.classification

import breeze.linalg.Vector
import com.linkedin.mlease.spark.function.TwiceDiffFunction
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.{LogisticLossFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.optimization.{LBFGS, RegularizationContext}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearAlgorithm
import com.linkedin.photon.ml.util.DataValidators

/**
 * Train a classification model for Logistic Regression.
 * @note Labels used in Logistic Regression should be {0, 1}
 * @author xazhang
 * @author dpeng
 */
class LogisticRegressionAlgorithm extends GeneralizedLinearAlgorithm[LogisticRegressionModel, TwiceDiffFunction[LabeledPoint]] {

  override protected val validators = List(DataValidators.finiteFeaturesValidator, DataValidators.binaryLabelValidator)

  /**
   *  Objective function = loss function + l2weight * regularization
   *  Only the L2 regularization part is implemented in the objective function. L1 part is implemented through the optimizer. See [[LBFGS]].
   */
  override protected def createObjectiveFunction(regularizationContext: RegularizationContext, regularizationWeight: Double): TwiceDiffFunction[LabeledPoint] = {
    TwiceDiffFunction.withRegularization(new LogisticLossFunction, regularizationContext, regularizationWeight)
  }

  /**
   * Create a logistic regression model given the estimated coefficients and intercept
   */
  override protected def createModel(coefficients: Vector[Double], intercept: Option[Double]) = {
    new LogisticRegressionModel(coefficients, intercept)
  }
}
