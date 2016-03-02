package com.linkedin.photon.ml.algorithm

import scala.collection.Set

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.{KeyValueScore, RandomEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.{Coefficients, RandomEffectModel, Model}
import com.linkedin.photon.ml.optimization.game.{
  RandomEffectOptimizationTracker, OptimizationTracker, RandomEffectOptimizationProblem}

/**
 * The optimization problem coordinate for a random effect model
 *
 * @param randomEffectDataSet the training dataset
 * @param randomEffectOptimizationProblem the random effect optimization problem
 * @author xazhang
 */
abstract class RandomEffectCoordinate[F <: TwiceDiffFunction[LabeledPoint], R <: RandomEffectCoordinate[F, R]](
    randomEffectDataSet: RandomEffectDataSet,
    randomEffectOptimizationProblem: RandomEffectOptimizationProblem[F])
  extends Coordinate[RandomEffectDataSet, RandomEffectCoordinate[F, R]](randomEffectDataSet) {

  /**
   * Score the model
   *
   * @param model the model to score
   * @return scores
   */
  override def score(model: Model): KeyValueScore = {
    model match {
      case randomEffectModel: RandomEffectModel => RandomEffectCoordinate.score(randomEffectDataSet, randomEffectModel)
      case _ => throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
          s"in ${this.getClass} is not supported!")
    }
  }

  /**
   * Update the model
   *
   * @param model the model to update
   */
  override protected def updateModel(model: Model): (Model, OptimizationTracker) = {
    model match {
      case randomEffectModel: RandomEffectModel =>
        RandomEffectCoordinate.updateModel(randomEffectDataSet, randomEffectOptimizationProblem, randomEffectModel)
      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} " +
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
      case randomEffectModel: RandomEffectModel =>
        randomEffectOptimizationProblem.getRegularizationTermValue(randomEffectModel.coefficientsRDD)
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
  override protected def updateCoordinateWithDataSet(
      updatedRandomEffectDataSet: RandomEffectDataSet): RandomEffectCoordinate[F, R] = {
    updateRandomEffectCoordinateWithDataSet(updatedRandomEffectDataSet)
  }

  /**
   * Update the coordinate with a dataset. Subclasses should implement this to supply the appropriate update.
   *
   * @param updatedRandomEffectDataSet the updated dataset
   */
  protected def updateRandomEffectCoordinateWithDataSet(updatedRandomEffectDataSet: RandomEffectDataSet): R
}

object RandomEffectCoordinate {

  /**
   * Score the model
   *
   * @param randomEffectDataSet the dataset
   * @param randomEffectModel the model
   * @return scores
   */
  protected[algorithm] def score(
      randomEffectDataSet: RandomEffectDataSet,
      randomEffectModel: RandomEffectModel) : KeyValueScore = {

    val activeScores = randomEffectDataSet.activeData.join(randomEffectModel.coefficientsRDD)
        .flatMap { case (individualId, (localDataSet, coefficients)) =>
      localDataSet.dataPoints.map { case (globalId, labeledPoint) =>
        (globalId, coefficients.computeScore(labeledPoint.features))
      }
    }.partitionBy(randomEffectDataSet.globalIdPartitioner)
        .setName("Active scores")
        .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val passiveDataOption = randomEffectDataSet.passiveDataOption
    if (passiveDataOption.isDefined) {
      val passiveDataIndividualIdsOption = randomEffectDataSet.passiveDataIndividualIdsOption
      val passiveScores = computePassiveScores(passiveDataOption.get, passiveDataIndividualIdsOption.get,
        randomEffectModel.coefficientsRDD)
          .setName("Passive scores")
          .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

      new KeyValueScore(activeScores ++ passiveScores)
    } else {
      new KeyValueScore(activeScores)
    }
  }

  /**
   * Computes passive scores
   * TODO: Explain passive data
   *
   * @param passiveData the dataset
   * @param passiveDataIndividualIds the set of individual random effect entity ids
   * @param coefficientsRDD model coefficients
   * @return scores
   */
  private def computePassiveScores(
      passiveData: RDD[(Long, (String, LabeledPoint))],
      passiveDataIndividualIds: Broadcast[Set[String]],
      coefficientsRDD: RDD[(String, Coefficients)]): RDD[(Long, Double)] = {

    val modelsForPassiveData = coefficientsRDD.filter { case (shardId, _) =>
      passiveDataIndividualIds.value.contains(shardId)
    }.collectAsMap()

    //TODO: Need a better design that properly unpersists the broadcasted variables and persists the computed RDD
    val modelsForPassiveDataBroadcast = passiveData.sparkContext.broadcast(modelsForPassiveData)
    val passiveScores = passiveData.mapValues { case (individualId, labeledPoint) =>
      modelsForPassiveDataBroadcast.value(individualId).computeScore(labeledPoint.features)
    }

    passiveScores.setName("passive scores").persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL).count()
    modelsForPassiveDataBroadcast.unpersist()

    passiveScores
  }

  /**
   * Update the model (i.e. run the coordinate optimizer)
   *
   * @param randomEffectDataSet the dataset
   * @param randomEffectOptimizationProblem the optimization problem
   * @param randomEffectModel the model
   * @return tuple of updated model and optimization tracker
   */
  protected[algorithm] def updateModel[F <: TwiceDiffFunction[LabeledPoint]](
      randomEffectDataSet: RandomEffectDataSet,
      randomEffectOptimizationProblem: RandomEffectOptimizationProblem[F],
      randomEffectModel: RandomEffectModel) : (RandomEffectModel, RandomEffectOptimizationTracker) = {

    val result = randomEffectDataSet.activeData
      .join(randomEffectOptimizationProblem.optimizationProblems)
      .join(randomEffectModel.coefficientsRDD)
      .mapValues {
        case (((localDataSet, optimizationProblem), localModel)) =>
          val trainingLabeledPoints = localDataSet.dataPoints.map(_._2)
          val (updatedLocalModel, _) = optimizationProblem.updatedCoefficientsMeans(trainingLabeledPoints, localModel)

          (updatedLocalModel, optimizationProblem)
      }
      .setName(s"Tmp updated random effect algorithm results")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val updatedRandomEffectModel = randomEffectModel.updateRandomEffectModel(result.mapValues(_._1))
        .setName(s"Updated random effect model")
        .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
        .materialize()

    val optimizationStateTrackers = result.values.map(_._2.optimizer.getStateTracker.get)
    val optimizationTracker = new RandomEffectOptimizationTracker(optimizationStateTrackers)
        .setName(s"Random effect optimization tracker")
        .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
        .materialize()

    //safely unpersist the RDDs after their dependencies are all materialized
    result.unpersist()

    (updatedRandomEffectModel, optimizationTracker)
  }
}
