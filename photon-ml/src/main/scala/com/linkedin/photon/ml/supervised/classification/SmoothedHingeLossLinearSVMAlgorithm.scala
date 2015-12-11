package com.linkedin.photon.ml.supervised.classification

import breeze.linalg.Vector
import com.linkedin.photon.ml.data.{ObjectProvider, LabeledPoint}
import com.linkedin.photon.ml.function.{SmoothedHingeLossFunction, DiffFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.RegularizationContext
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearAlgorithm
import com.linkedin.photon.ml.util.DataValidators
import org.apache.spark.rdd.RDD

/**
 * Approximate linear SVM via soft hinge loss
 */
class SmoothedHingeLossLinearSVMAlgorithm
  extends GeneralizedLinearAlgorithm[SmoothedHingeLossLinearSVMModel, DiffFunction[LabeledPoint]] {

  override protected val validators: Seq[RDD[LabeledPoint] => Boolean] =
    List(DataValidators.logisticRegressionValidator)

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
    DiffFunction.withRegularization(new SmoothedHingeLossFunction(), regularizationContext, regularizationWeight)
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
