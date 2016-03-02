package com.linkedin.photon.ml.algorithm


import com.linkedin.photon.ml.data.{
  KeyValueScore, RandomEffectDataSet, LabeledPoint, RandomEffectDataSetInProjectedSpace}
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.{RandomEffectModelInProjectedSpace, Coefficients, RandomEffectModel, Model}
import com.linkedin.photon.ml.optimization.game.{OptimizationTracker, RandomEffectOptimizationProblem}


/**
 * The optimization problem coordinate for a random effect model in projected space
 *
 * @param randomEffectDataSetInProjectedSpace the training dataset
 * @param randomEffectOptimizationProblem the fixed effect optimization problem
 * @author xazhang
 */
class RandomEffectCoordinateInProjectedSpace[F <: TwiceDiffFunction[LabeledPoint]](
    randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace,
    randomEffectOptimizationProblem: RandomEffectOptimizationProblem[F])
  extends RandomEffectCoordinate[F, RandomEffectCoordinateInProjectedSpace[F]](
    randomEffectDataSetInProjectedSpace, randomEffectOptimizationProblem) {

  /**
   * Initialize the model
   *
   * @param seed random seed
   */
  override def initializeModel(seed: Long): Model = {
    RandomEffectCoordinateInProjectedSpace.initializeZeroModel(randomEffectDataSetInProjectedSpace)
  }

  /**
   * Update the model (i.e. run the coordinate optimizer)
   *
   * @param model the model
   * @return tuple of updated model and optimization tracker
   */
  override protected def updateModel(model: Model): (Model, OptimizationTracker) = {
    model match {
      case randomEffectModelWithProjector: RandomEffectModelInProjectedSpace =>
        val randomEffectModel = randomEffectModelWithProjector.toRandomEffectModel
        val (updatedModel, optimizationTracker) = super.updateModel(randomEffectModel)
        val updatedCoefficientsRDD = updatedModel.asInstanceOf[RandomEffectModel].coefficientsRDD
        val updatedRandomEffectModelWithProjector =
          randomEffectModelWithProjector.updateRandomEffectModelInProjectedSpace(updatedCoefficientsRDD)
        (updatedRandomEffectModelWithProjector, optimizationTracker)
      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} in ${this.getClass} is " +
            s"not supported!")
    }
  }

  /**
   * Score the model
   *
   * @param model the model to score
   * @return scores
   */
  override def score(model: Model): KeyValueScore = {
    model match {
      case randomEffectModelWithProjector: RandomEffectModelInProjectedSpace =>
        val randomEffectModel = randomEffectModelWithProjector.toRandomEffectModel
        super.score(randomEffectModel)
      case _ =>
        throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
            s"in ${this.getClass} is not supported!")
    }
  }

  /**
   * Compute the regularization term value
   *
   * @param model the model
   * @return regularization term value
   */
  override def computeRegularizationTermValue(model: Model): Double = {
    model match {
      case randomEffectModelWithProjector: RandomEffectModelInProjectedSpace =>
        val randomEffectModel = randomEffectModelWithProjector.toRandomEffectModel
        super.computeRegularizationTermValue(randomEffectModel)
      case _ =>
        throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
            s"type ${model.getClass} in ${this.getClass} is not supported!")
    }
  }

  /**
   * Update the coordinate with a dataset
   *
   * @param updatedRandomEffectDataSet the updated dataset
   * @return the updated coordinate
   */
  override protected def updateRandomEffectCoordinateWithDataSet(
      updatedRandomEffectDataSet: RandomEffectDataSet) : RandomEffectCoordinateInProjectedSpace[F] = {

    val updatedRandomEffectDataSetInProjectedSpace = new RandomEffectDataSetInProjectedSpace(updatedRandomEffectDataSet,
      randomEffectDataSetInProjectedSpace.randomEffectProjector)
    new RandomEffectCoordinateInProjectedSpace(updatedRandomEffectDataSetInProjectedSpace,
      randomEffectOptimizationProblem)
  }
}

object RandomEffectCoordinateInProjectedSpace {

  /**
   * Initialize a zero model
   *
   * @param randomEffectDataSetInProjectedSpace the dataset
   */
  private def initializeZeroModel(
      randomEffectDataSetInProjectedSpace: RandomEffectDataSetInProjectedSpace) : RandomEffectModelInProjectedSpace = {

    val randomEffectModel = randomEffectDataSetInProjectedSpace.activeData.mapValues(localDataSet =>
      Coefficients.initializeZeroCoefficients(localDataSet.numFeatures)
    )
    val randomEffectId = randomEffectDataSetInProjectedSpace.randomEffectId
    val featureShardId = randomEffectDataSetInProjectedSpace.featureShardId
    val randomEffectProjector = randomEffectDataSetInProjectedSpace.randomEffectProjector
    new RandomEffectModelInProjectedSpace(randomEffectModel, randomEffectProjector, randomEffectId, featureShardId)
  }
}
