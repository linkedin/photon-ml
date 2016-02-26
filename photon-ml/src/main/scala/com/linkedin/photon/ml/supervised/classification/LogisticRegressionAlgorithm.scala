package com.linkedin.photon.ml.supervised.classification

import breeze.linalg.Vector
import com.linkedin.photon.ml.data.{LabeledPoint, ObjectProvider}
import com.linkedin.photon.ml.function.{LogisticLossFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.{LBFGS, Optimizer, OptimizerConfig, OptimizerFactory, RegularizationContext}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearAlgorithm

/**
 * Train a classification model for Logistic Regression.
 * @note Labels used in Logistic Regression should be {0, 1}
 * @author xazhang
 * @author dpeng
 */
class LogisticRegressionAlgorithm
  extends GeneralizedLinearAlgorithm[LogisticRegressionModel, TwiceDiffFunction[LabeledPoint]] {

  /**
   *  Objective function = loss function + l2weight * regularization
   *  Only the L2 regularization part is implemented in the objective function. L1 part is implemented through the
   *  optimizer. See [[LBFGS]].
   */
  override protected def createObjectiveFunction(
      normalizationContext: ObjectProvider[NormalizationContext],
      regularizationContext: RegularizationContext,
      regularizationWeight: Double): TwiceDiffFunction[LabeledPoint] = {
    val basicFunction = new LogisticLossFunction(normalizationContext)
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
   * Create a logistic regression model given the estimated coefficients and intercept
   */
  override protected def createModel(coefficients: Vector[Double], intercept: Option[Double]) = {
    new LogisticRegressionModel(coefficients, intercept)
  }
}
