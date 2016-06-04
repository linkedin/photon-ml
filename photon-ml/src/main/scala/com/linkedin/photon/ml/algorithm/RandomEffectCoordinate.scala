/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.algorithm

import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.{KeyValueScore, LabeledPoint, RandomEffectDataSet}
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.{Coefficients, DatumScoringModel, RandomEffectModel}
import com.linkedin.photon.ml.optimization.game.{OptimizationTracker, RandomEffectOptimizationProblem, RandomEffectOptimizationTracker}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.Set

/**
  * The optimization problem coordinate for a random effect model
  *
  * @param randomEffectDataSet The training dataset
  * @param randomEffectOptimizationProblem The random effect optimization problem
  */
protected[ml] abstract class RandomEffectCoordinate[F <: TwiceDiffFunction[LabeledPoint]](
    randomEffectDataSet: RandomEffectDataSet,
    randomEffectOptimizationProblem: RandomEffectOptimizationProblem[F])
  extends Coordinate[RandomEffectDataSet, RandomEffectCoordinate[F]](randomEffectDataSet) {

  /**
    * Score the model
    *
    * @param model The model to score
    * @return Scores
    */
  protected[algorithm] override def score(model: DatumScoringModel): KeyValueScore = {
    model match {
      case randomEffectModel: RandomEffectModel => RandomEffectCoordinate.score(randomEffectDataSet, randomEffectModel)
      case _ => throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
          s"in ${this.getClass} is not supported!")
    }
  }

  /**
    * Update the model
    *
    * @param model The model to update
    */
  protected[algorithm] override def updateModel(model: DatumScoringModel): (DatumScoringModel, OptimizationTracker) = {
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
    * @param model The model
    * @return Regularization term value
    */
  protected[algorithm] override def computeRegularizationTermValue(model: DatumScoringModel): Double = model match {
    case randomEffectModel: RandomEffectModel =>
      randomEffectOptimizationProblem.getRegularizationTermValue(randomEffectModel.coefficientsRDD)

    case _ =>
      throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
        s"type ${model.getClass} in ${this.getClass} is not supported!")
  }
}

object RandomEffectCoordinate {

  /**
    * Score the model
    *
    * @param randomEffectDataSet The dataset
    * @param randomEffectModel The model
    * @return Scores
    */
  protected[algorithm] def score(randomEffectDataSet: RandomEffectDataSet, randomEffectModel: RandomEffectModel)
    : KeyValueScore = {

    val activeScores = randomEffectDataSet
      .activeData
      .join(randomEffectModel.coefficientsRDD)
      .flatMap { case (individualId, (localDataSet, coefficients)) =>
        localDataSet.dataPoints.map { case (globalId, labeledPoint) =>
          (globalId, coefficients.computeScore(labeledPoint.features))
        }
      }
      .partitionBy(randomEffectDataSet.globalIdPartitioner)
      .setName("Active scores")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val passiveDataOption = randomEffectDataSet.passiveDataOption
    if (passiveDataOption.isDefined) {
      val passiveDataIndividualIdsOption = randomEffectDataSet.passiveDataIndividualIdsOption
      val passiveScores = computePassiveScores(
          passiveDataOption.get,
          passiveDataIndividualIdsOption.get,
          randomEffectModel.coefficientsRDD)
        .setName("Passive scores")
        .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

      new KeyValueScore(activeScores ++ passiveScores)
    } else {
      new KeyValueScore(activeScores)
    }
  }

  // TODO: Explain passive data
  /**
    * Computes passive scores
    *
    * @param passiveData The dataset
    * @param passiveDataIndividualIds The set of individual random effect entity ids
    * @param coefficientsRDD Model coefficients
    * @return Scores
    */
  private def computePassiveScores(
      passiveData: RDD[(Long, (String, LabeledPoint))],
      passiveDataIndividualIds: Broadcast[Set[String]],
      coefficientsRDD: RDD[(String, Coefficients)]): RDD[(Long, Double)] = {

    val modelsForPassiveData = coefficientsRDD
      .filter { case (shardId, _) =>
        passiveDataIndividualIds.value.contains(shardId)
      }
      .collectAsMap()

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
    * @param randomEffectDataSet The dataset
    * @param randomEffectOptimizationProblem The optimization problem
    * @param randomEffectModel The model
    * @return Tuple of updated model and optimization tracker
    */
  protected[algorithm] def updateModel[F <: TwiceDiffFunction[LabeledPoint]](
      randomEffectDataSet: RandomEffectDataSet,
      randomEffectOptimizationProblem: RandomEffectOptimizationProblem[F],
      randomEffectModel: RandomEffectModel) : (RandomEffectModel, RandomEffectOptimizationTracker) = {

    val result = randomEffectDataSet
      .activeData
      .join(randomEffectOptimizationProblem.optimizationProblems)
      .join(randomEffectModel.coefficientsRDD)
      .mapValues {
        case (((localDataSet, optimizationProblem), localModel)) =>
          val trainingLabeledPoints = localDataSet.dataPoints.map(_._2)
          val (updatedLocalModel, _) = optimizationProblem.updateCoefficientMeans(trainingLabeledPoints, localModel)

          (updatedLocalModel, optimizationProblem)
      }
      .setName(s"Tmp updated random effect algorithm results")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val updatedRandomEffectModel = randomEffectModel
      .updateRandomEffectModel(result.mapValues(_._1))
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
