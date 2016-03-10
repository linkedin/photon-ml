package com.linkedin.photon.ml.optimization.game

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.sampler.{DefaultDownSampler, BinaryClassificationDownSampler, DownSampler}
import com.linkedin.photon.ml.supervised.TaskType._

/**
 * Components of an optimization problem
 *
 * @param optimizer The underlying optimizer who does the job
 * @param objectiveFunction The objective function upon which to optimize
 * @param lossFunction The loss function of the optimization problem
 * @param regularizationWeight The regularization weight of the optimization problem
 * @param sampler The sampler used to down-sample the training data points
 * @tparam F The type of objective/loss function
 * @author xazhang
 */
case class OptimizationProblem[F <: TwiceDiffFunction[LabeledPoint]](
    optimizer: AbstractOptimizer[LabeledPoint, F],
    objectiveFunction: F,
    lossFunction: F,
    regularizationWeight: Double,
    sampler: DownSampler) {

  /**
   * Compute the regularization term value
   *
   * @param model the model
   * @return regularization term value
   */
  def getRegularizationTermValue(model: Coefficients): Double = {
    //TODO: L1 regularization?
    val coefficients = model.means
    coefficients.dot(coefficients) * regularizationWeight / 2
  }

  /**
   * Update coefficient variances
   *
   * @param labeledPoints the training dataset
   * @param previousModel the previous model
   * @return updated coefficients
   */
  def updateCoefficientsVariances(labeledPoints: Iterable[LabeledPoint], previousModel: Coefficients): Coefficients = {
    val updatedVariance = objectiveFunction.hessianDiagonal(labeledPoints, previousModel.means)
        .map(v => 1.0 / (v + MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD))
    Coefficients(previousModel.means, Some(updatedVariance))
  }

  /**
   * Update coefficient means (i.e., optimize)
   *
   * @param labeledPoints the training dataset
   * @param previousModel the previous model
   * @return updated coefficients
   */
  def updateCoefficientMeans(
      labeledPoints: Iterable[LabeledPoint],
      previousModel: Coefficients): (Coefficients, Double) = {

    val (updatedCoefficients, loss) = optimizer.optimize(labeledPoints, objectiveFunction, previousModel.means)
    (Coefficients(updatedCoefficients, previousModel.variancesOption), loss)
  }

  /**
   * Update coefficient variances
   *
   * @param labeledPoints the training dataset
   * @param previousModel the previous model
   * @return updated coefficients
   */
  def updateCoefficientsVariances(
      labeledPoints: RDD[LabeledPoint],
      previousModel: Coefficients): Coefficients = {

    //TODO: unpersist the broadcasted updatedCoefficients after updatedVariance is computed
    val broadcastedUpdatedCoefficients = labeledPoints.sparkContext.broadcast(previousModel.means)
    val updatedVariance = objectiveFunction.hessianDiagonal(labeledPoints, broadcastedUpdatedCoefficients)
        .map(v => 1.0 / (v + MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD))
    Coefficients(previousModel.means, Some(updatedVariance))
  }

  /**
   * Update coefficient means (i.e., optimize)
   *
   * @param labeledPoints the training dataset
   * @param previousModel the previous model
   * @return updated coefficients
   */
  def updateCoefficientMeans(
      labeledPoints: RDD[LabeledPoint],
      previousModel: Coefficients): (Coefficients, Double) = {

    val (updatedCoefficients, loss) = optimizer.optimize(labeledPoints, objectiveFunction, previousModel.means)
    (Coefficients(updatedCoefficients, previousModel.variancesOption), loss)
  }
}

object OptimizationProblem {

  /**
   * Build an optimization problem instance
   *
   * @param taskType the task type (e.g. LinearRegression, LogisticRegression)
   * @param configuration optimization configuration
   * @return optimization problem instance
   * @todo build optimization problem with more general type of functions
   */
  def buildOptimizationProblem(
      taskType: TaskType,
      configuration: GLMOptimizationConfiguration): OptimizationProblem[TwiceDiffFunction[LabeledPoint]] = {

    val maxNumberIterations = configuration.maxNumberIterations
    val convergenceTolerance = configuration.convergenceTolerance
    val regularizationWeight = configuration.regularizationWeight
    val downSamplingRate = configuration.downSamplingRate
    val optimizerType = configuration.optimizerType
    val regularizationType = configuration.regularizationType

    val lossFunction = taskType match {
      case LOGISTIC_REGRESSION => new LogisticLossFunction
      case LINEAR_REGRESSION => new SquaredLossFunction
      case _ => throw new Exception(s"Loss function for taskType $taskType is currently not supported.")
    }

    val objectiveFunction = regularizationType match {
      case RegularizationType.L2 => TwiceDiffFunction.withL2Regularization(lossFunction, regularizationWeight)
      case RegularizationType.L1 => TwiceDiffFunction.withL1Regularization(lossFunction, regularizationWeight)
      case other => throw new UnsupportedOperationException(s"Regularization of type $other is not supported.")
    }

    val optimizer = optimizerType match {
      case OptimizerType.LBFGS =>
        new LBFGS[LabeledPoint]
      case OptimizerType.TRON =>
        if (regularizationType == RegularizationType.L2) new TRON[LabeledPoint]
        else throw new IllegalArgumentException(s"For regularization of type $regularizationType, optimizer of " +
            s"type ${OptimizerType.TRON} is not supported!")
      case any =>
        throw new UnsupportedOperationException(s"Optimizer of type $any is not supported")
    }

    // For warm start model training
    optimizer.isReusingPreviousInitialState = true

    val sampler = taskType match {
      case LOGISTIC_REGRESSION => new BinaryClassificationDownSampler(downSamplingRate)
      case LINEAR_REGRESSION => new DefaultDownSampler(downSamplingRate)
      case _ => throw new Exception(s"Sampler for taskType $taskType is currently not supported.")
    }

    optimizer.setMaximumIterations(maxNumberIterations)
    optimizer.setTolerance(convergenceTolerance)

    OptimizationProblem(optimizer, objectiveFunction, lossFunction, regularizationWeight, sampler)
  }
}