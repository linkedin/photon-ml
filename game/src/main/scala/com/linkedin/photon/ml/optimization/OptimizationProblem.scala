package com.linkedin.photon.ml.optimization

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.contants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.sampler.{DefaultSampler, BinaryClassificationSampler, Sampler}
import com.linkedin.photon.ml.supervised.TaskType._


/**
 * Components of an optimization problem
 * @param optimizer The underlying optimizer who does the job
 * @param objectiveFunction The objective function upon which to optimize
 * @param lossFunction The loss function of the optimization problem
 * @param regularizationWeight The regularization weight of the optimization problem
 * @param sampler The sampler used to down-sample the training data points
 * @tparam F The type of objective/loss function
 * @author xazhang
 */
case class OptimizationProblem[F <: EnhancedTwiceDiffFunction[LabeledPoint]](
    optimizer: AbstractOptimizer[LabeledPoint, F],
    objectiveFunction: F,
    lossFunction: F,
    regularizationWeight: Double,
    sampler: Sampler) {

  def getRegularizationTermValue(model: Coefficients): Double = {
    //TODO: L1 regularization?
    val coefficients = model.means
    coefficients.dot(coefficients) * regularizationWeight / 2
  }

  def updateCoefficientsVariances(labeledPoints: Iterable[LabeledPoint], previousModel: Coefficients): Coefficients = {
    val updatedVariance = objectiveFunction.hessianDiagonal(labeledPoints, previousModel.means)
        .map(v => 1.0 / (v + MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD))
    Coefficients(previousModel.means, Some(updatedVariance))
  }

  def updatedCoefficientsMeans(labeledPoints: Iterable[LabeledPoint], previousModel: Coefficients)
  : (Coefficients, Double) = {

    val (updatedCoefficients, loss) = optimizer.optimize(labeledPoints, objectiveFunction, previousModel.means)
    (Coefficients(updatedCoefficients, previousModel.variancesOption), loss)
  }

  def updateCoefficientsVariances(labeledPoints: RDD[LabeledPoint], previousModel: Coefficients): Coefficients = {
    //TODO: unpersist the broadcasted updatedCoefficients after updatedVariance is computed
    val broadcastedUpdatedCoefficients = labeledPoints.sparkContext.broadcast(previousModel.means)
    val updatedVariance = objectiveFunction.hessianDiagonal(labeledPoints, broadcastedUpdatedCoefficients)
        .map(v => 1.0 / (v + MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD))
    Coefficients(previousModel.means, Some(updatedVariance))
  }

  def updatedCoefficientsMeans(labeledPoints: RDD[LabeledPoint], previousModel: Coefficients)
  : (Coefficients, Double) = {

    val (updatedCoefficients, loss) = optimizer.optimize(labeledPoints, objectiveFunction, previousModel.means)
    (Coefficients(updatedCoefficients, previousModel.variancesOption), loss)
  }
}

object OptimizationProblem {

  //TODO: build optimization problem with more general type of functions
  def buildOptimizationProblem(taskType: TaskType, configuration: GLMOptimizationConfiguration)
  : OptimizationProblem[EnhancedTwiceDiffFunction[LabeledPoint]] = {

    val maxNumberIterations = configuration.maxNumberIterations
    val convergenceTolerance = configuration.convergenceTolerance
    val regularizationWeight = configuration.regularizationWeight
    val downSamplingRate = configuration.downSamplingRate
    val optimizerType = configuration.optimizerType
    val regularizationType = configuration.regularizationType
    val lossFunction = taskType match {
      case LOGISTIC_REGRESSION => new EnhancedLogisticLossFunction
      case LINEAR_REGRESSION => new EnhancedSquaredLossFunction
      case _ => throw new Exception(s"Loss function for taskType $taskType is currently not supported.")
    }
    val objectiveFunction = regularizationType match {
      case RegularizationType.L2 => EnhancedTwiceDiffFunction.withL2Regularization(lossFunction, regularizationWeight)
      case RegularizationType.L1 => EnhancedTwiceDiffFunction.withL1Regularization(lossFunction, regularizationWeight)
      case other => throw new UnsupportedOperationException(s"Regularization of type $other is not supported.")
    }
    val optimizer = optimizerType match {
      case OptimizerType.LBFGS =>
        new LBFGS[LabeledPoint]
      case OptimizerType.TRON =>
        if (regularizationType == RegularizationType.L2) new TRON[LabeledPoint] else new LBFGS[LabeledPoint]
      case any =>
        throw new UnsupportedOperationException(s"optimizer of type $any is not supported")
    }
    val sampler = taskType match {
      case LOGISTIC_REGRESSION => new BinaryClassificationSampler(downSamplingRate)
      case LINEAR_REGRESSION => new DefaultSampler(downSamplingRate)
      case _ => throw new Exception(s"Sampler for taskType $taskType is currently not supported.")
    }
    optimizer.setMaximumIterations(maxNumberIterations)
    optimizer.setTolerance(convergenceTolerance)
    OptimizationProblem(optimizer, objectiveFunction, lossFunction, regularizationWeight, sampler)
  }
}
