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
package com.linkedin.photon.ml.model

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.data.{GameDatum, KeyValueScore}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD._
import org.apache.spark.storage.StorageLevel

/**
  * Representation of a random effect model
  *
  * @param coefficientsRDD The coefficients
  * @param randomEffectId The random effect type id
  * @param featureShardId The feature shard id
  */
protected[ml] class RandomEffectModel(
    val coefficientsRDD: RDD[(String, Coefficients)],
    val randomEffectId: String,
    val featureShardId: String)
  extends DatumScoringModel with RDDLike {

  override def sparkContext: SparkContext = coefficientsRDD.sparkContext

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!coefficientsRDD.getStorageLevel.isValid) coefficientsRDD.persist(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    if (coefficientsRDD.getStorageLevel.isValid) coefficientsRDD.unpersist()
    this
  }

  override def setName(name: String): this.type = {
    coefficientsRDD.setName(name)
    this
  }

  override def materialize(): this.type = {
    coefficientsRDD.count()
    this
  }

  /**
    * Compute the score for the dataset
    *
    * @param dataPoints The dataset
    * @return The score
    */
  override def score(dataPoints: RDD[(Long, GameDatum)]): KeyValueScore =
    RandomEffectModel.score(dataPoints, coefficientsRDD, randomEffectId, featureShardId)

  /**
    * Build a summary string for the model
    *
    * @return String representation
    */
  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(s"Random effect model of randomEffectId $randomEffectId, " +
        s"featureShardId $featureShardId summary:")
    stringBuilder.append(s"\nLength: ${coefficientsRDD.values.map(_.means.activeSize).stats()}")
    stringBuilder.append(s"\nMean: ${coefficientsRDD.map(_._2.meansL2Norm).stats()}")
    if (coefficientsRDD.first()._2.variancesOption.isDefined) {
      stringBuilder.append(s"\nvariance: ${coefficientsRDD.map(_._2.variancesL2NormOption.get).stats()}")
    }
    stringBuilder.toString()
  }

  /**
    * Update the random effect model with new underlying models
    *
    * @param updatedModelsRdd The new underlying models, one per individual
    * @return The updated random effect model
    */
  def updateRandomEffectModel(updatedModelsRdd: RDD[(String, Coefficients)]): RandomEffectModel =
    new RandomEffectModel(updatedModelsRdd, randomEffectId, featureShardId)

  override def equals(that: Any): Boolean = {
    that match {
      case other: RandomEffectModel =>
        val sameMetaData = this.randomEffectId == other.randomEffectId && this.featureShardId == other.featureShardId
        val sameCoefficientsRDD = this
          .coefficientsRDD
          .fullOuterJoin(other.coefficientsRDD)
          .mapPartitions(iterator =>
            Iterator.single(iterator.forall { case (_, (coefficientsOpt1, coefficientsOpt2)) =>
              coefficientsOpt1.isDefined &&
                coefficientsOpt2.isDefined &&
                coefficientsOpt1.get.equals(coefficientsOpt2.get)
            }))
          .filter(!_)
          .count() == 0

        sameMetaData && sameCoefficientsRDD
      case _ => false
    }
  }

  // TODO: Violation of the hashCode() contract
  override def hashCode(): Int = super.hashCode()
}

object RandomEffectModel {

  /**
    * Compute the score for the dataset
    *
    * @param dataPoints The dataset to score
    * @param coefficientsRDD The models to use for scoring
    * @param randomEffectId The random effect type id
    * @param featureShardId The feature shard id
    * @return The scores
    */
  protected def score(
      dataPoints: RDD[(Long, GameDatum)],
      coefficientsRDD: RDD[(String, Coefficients)],
      randomEffectId: String,
      featureShardId: String): KeyValueScore = {

    val scores = dataPoints
      .map { case (globalId, gameData) =>
        val individualId = gameData.randomEffectIdToIndividualIdMap(randomEffectId)
        val features = gameData.featureShardContainer(featureShardId)
        (individualId, (globalId, features))
      }
      .cogroup(coefficientsRDD)
      .flatMap { case (individualId, (globalIdAndFeaturesIterable, coefficientsIterable)) =>
        assert(coefficientsIterable.size <= 1,
          s"More than one model (${coefficientsIterable.size}) found for individual Id $individualId of " +
            s"random effect Id $randomEffectId")

        if (coefficientsIterable.isEmpty) {
          globalIdAndFeaturesIterable.map { case (globalId, _) => (globalId, 0.0) }
        } else {
          val coefficients = coefficientsIterable.head
          globalIdAndFeaturesIterable.map { case (globalId, features) =>
            (globalId, coefficients.computeScore(features))
          }
        }
      }

    new KeyValueScore(scores)
  }
}
