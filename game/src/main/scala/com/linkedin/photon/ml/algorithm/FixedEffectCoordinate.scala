package com.linkedin.photon.ml.algorithm


import com.linkedin.photon.ml.contants.StorageLevel
import com.linkedin.photon.ml.data.{KeyValueScore, FixedEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.function.EnhancedTwiceDiffFunction
import com.linkedin.photon.ml.model.{Coefficients, FixedEffectModel, Model}
import com.linkedin.photon.ml.optimization.{FixedEffectOptimizationTracker, OptimizationTracker, OptimizationProblem}
import com.linkedin.photon.ml.util.PhotonLogger


/**
 * @author xazhang
 */
class FixedEffectCoordinate [F <: EnhancedTwiceDiffFunction[LabeledPoint]](
    fixedEffectDataSet: FixedEffectDataSet,
    private var optimizationProblem: OptimizationProblem[F])
    extends Coordinate[FixedEffectDataSet, FixedEffectCoordinate[F]](fixedEffectDataSet) {

  def initializeModel(seed: Long): FixedEffectModel = {
    FixedEffectCoordinate.initializeZeroModel(fixedEffectDataSet)
  }

  override protected def updateCoordinateWithDataSet(fixedEffectDataSet: FixedEffectDataSet)
  : FixedEffectCoordinate[F] = {
    new FixedEffectCoordinate[F](fixedEffectDataSet, optimizationProblem)
  }

  override protected def updateModel(model: Model): (Model, OptimizationTracker) = {
    model match {
      case fixedEffectModel: FixedEffectModel =>
        val (updatedFixedEffectModel, updatedOptimizationProblem) =
          FixedEffectCoordinate.updateModel(fixedEffectDataSet, optimizationProblem, fixedEffectModel)
        //Note that the optimizationProblem will memorize the current state of optimization,
        //and the next round of updating global models will share the same convergence criteria as this one.
        optimizationProblem = updatedOptimizationProblem
        val optimizationTracker = new FixedEffectOptimizationTracker(optimizationProblem.optimizer.getStateTracker.get)
        (updatedFixedEffectModel, optimizationTracker)
      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} in ${this.getClass} is " +
            s"not supported!")
    }
  }

  def score(model: Model): KeyValueScore = {
    model match {
      case fixedEffectModel: FixedEffectModel =>
        FixedEffectCoordinate.updateScore(fixedEffectDataSet, fixedEffectModel)
      case _ =>
        throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
            s"in ${this.getClass} is not supported!")
    }
  }

  def computeRegularizationTermValue(model: Model): Double = {
    model match {
      case fixedEffectModel: FixedEffectModel =>
        optimizationProblem.getRegularizationTermValue(fixedEffectModel.coefficients)
      case _ =>
        throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
            s"type ${model.getClass} in ${this.getClass} is not supported!")
    }
  }

  def summarize(logger: PhotonLogger): Unit = {
    logger.logDebug(s"Optimization stats: ${optimizationProblem.optimizer.getStateTracker.get}")
  }
}

object FixedEffectCoordinate {

  def initializeZeroModel(fixedEffectDataSet: FixedEffectDataSet): FixedEffectModel = {
    val numFeatures = fixedEffectDataSet.numFeatures
    val coefficients = Coefficients.initializeZeroCoefficients(numFeatures)
    val coefficientsBroadcast = fixedEffectDataSet.sparkContext.broadcast(coefficients)
    val featureShardId = fixedEffectDataSet.featureShardId
    new FixedEffectModel(coefficientsBroadcast, featureShardId)
  }

  private def updateModel[F <: EnhancedTwiceDiffFunction[LabeledPoint]](
      fixedEffectDataSet: FixedEffectDataSet,
      optimizationProblem: OptimizationProblem[F],
      fixedEffectModel: FixedEffectModel): (FixedEffectModel, OptimizationProblem[F]) = {

    val sampler = optimizationProblem.sampler
    val trainingData = sampler.downSample(fixedEffectDataSet.labeledPoints)
        .setName("In memory fixed effect training data set")
        .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    println(s"In memory fixed effect training data set summary:" +
        s"\n${new FixedEffectDataSet(trainingData, "").toSummaryString}")
    val coefficients = fixedEffectModel.coefficients
    val (updatedCoefficients, _) = optimizationProblem.updatedCoefficientsMeans(trainingData.values, coefficients)
    val updatedCoefficientsBroadcast = fixedEffectDataSet.sparkContext.broadcast(updatedCoefficients)
    val updatedFixedEffectModel = fixedEffectModel.update(updatedCoefficientsBroadcast)
    trainingData.unpersist()
    (updatedFixedEffectModel, optimizationProblem)
  }

  private def updateScore(fixedEffectDataSet: FixedEffectDataSet, fixedEffectModel: FixedEffectModel): KeyValueScore = {
    val coefficientsBroadcast = fixedEffectModel.coefficientsBroadcast
    val scores = fixedEffectDataSet.labeledPoints.mapValues { case LabeledPoint(_, features, _, _) =>
      coefficientsBroadcast.value.computeScore(features)
    }
    new KeyValueScore(scores)
  }
}
