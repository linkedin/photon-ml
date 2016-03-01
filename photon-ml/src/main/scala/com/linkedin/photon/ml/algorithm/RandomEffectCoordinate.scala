package com.linkedin.photon.ml.algorithm


import scala.collection.Set

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.contants.StorageLevel
import com.linkedin.photon.ml.data.{KeyValueScore, RandomEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.function.EnhancedTwiceDiffFunction
import com.linkedin.photon.ml.model.{Coefficients, RandomEffectModel, Model}
import com.linkedin.photon.ml.optimization.{RandomEffectOptimizationTracker, OptimizationTracker, RandomEffectOptimizationProblem}


/**
 * @author xazhang
 */
abstract class RandomEffectCoordinate[F <: EnhancedTwiceDiffFunction[LabeledPoint], R <:RandomEffectCoordinate[F, R] ](
    randomEffectDataSet: RandomEffectDataSet,
    randomEffectOptimizationProblem: RandomEffectOptimizationProblem[F])
    extends Coordinate[RandomEffectDataSet, RandomEffectCoordinate[F, R]](randomEffectDataSet) {

  override def score(model: Model): KeyValueScore = {
    model match {
      case randomEffectModel: RandomEffectModel => RandomEffectCoordinate.score(randomEffectDataSet, randomEffectModel)
      case _ => throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
          s"in ${this.getClass} is not supported!")
    }
  }

  override protected def updateModel(model: Model): (Model, OptimizationTracker) = {
    model match {
      case randomEffectModel: RandomEffectModel =>
        RandomEffectCoordinate.updateModel(randomEffectDataSet, randomEffectOptimizationProblem, randomEffectModel)
      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} " +
            s"in ${this.getClass} is not supported!")
    }
  }

  override def computeRegularizationTermValue(model: Model): Double = {
    model match {
      case randomEffectModel: RandomEffectModel =>
        randomEffectOptimizationProblem.getRegularizationTermValue(randomEffectModel.coefficientsRDD)
      case _ =>
        throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
            s"type ${model.getClass} in ${this.getClass} is not supported!")
    }
  }

  override protected def updateCoordinateWithDataSet(updatedRandomEffectDataSet: RandomEffectDataSet)
  : RandomEffectCoordinate[F, R] = {

    updateRandomEffectCoordinateWithDataSet(updatedRandomEffectDataSet)
  }

  protected def updateRandomEffectCoordinateWithDataSet(updatedRandomEffectDataSet: RandomEffectDataSet): R
}

object RandomEffectCoordinate {

  protected[algorithm] def score(randomEffectDataSet: RandomEffectDataSet, randomEffectModel: RandomEffectModel)
  : KeyValueScore = {

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

  protected[algorithm] def updateModel[F <: EnhancedTwiceDiffFunction[LabeledPoint]](
      randomEffectDataSet: RandomEffectDataSet,
      randomEffectOptimizationProblem: RandomEffectOptimizationProblem[F],
      randomEffectModel: RandomEffectModel)
  : (RandomEffectModel, RandomEffectOptimizationTracker) = {

    val result = randomEffectDataSet.activeData
        .join(randomEffectOptimizationProblem.optimizationProblems)
        .join(randomEffectModel.coefficientsRDD)
        .mapValues { case (((localDataSet, optimizationProblem), localModel)) =>
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
