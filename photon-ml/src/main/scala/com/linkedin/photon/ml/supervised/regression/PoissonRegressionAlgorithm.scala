package com.linkedin.photon.ml.supervised.regression

import breeze.linalg.Vector
import com.linkedin.photon.ml.data.{ObjectProvider, LabeledPoint}
import com.linkedin.photon.ml.function.{PoissonLossFunction, TwiceDiffFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.RegularizationContext
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearAlgorithm
import com.linkedin.photon.ml.util.DataValidators
import org.apache.spark.rdd.RDD


/**
 * Train a regression model using L2-regularized poisson regression.
 * @author asaha
 * @author dpeng
 */
class PoissonRegressionAlgorithm
  extends GeneralizedLinearAlgorithm[PoissonRegressionModel, TwiceDiffFunction[LabeledPoint]] {

  override protected val validators: Seq[RDD[LabeledPoint] => Boolean] = List(DataValidators.poissonRegressionValidator)

  /* Objective function = loss function + l2weight * regularization */
  override protected def createObjectiveFunction(
      normalizationContext: ObjectProvider[NormalizationContext],
      regularizationContext: RegularizationContext,
      regularizationWeight: Double): TwiceDiffFunction[LabeledPoint] = {
    TwiceDiffFunction.withRegularization(
      new PoissonLossFunction(normalizationContext), regularizationContext, regularizationWeight)
  }

  /**
   * Create a poisson regression model given the estimated coefficients and intercept
   */
  override protected def createModel(coefficients: Vector[Double], intercept: Option[Double]) = {
    new PoissonRegressionModel(coefficients, intercept)
  }
}
