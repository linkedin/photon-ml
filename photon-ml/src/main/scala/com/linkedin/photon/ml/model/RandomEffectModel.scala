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
 * @param modelsRDD The models
 * @param randomEffectType The random effect type
 * @param featureShardId The feature shard id
 */
protected[ml] class RandomEffectModel(
    val modelsRDD: RDD[(String, GeneralizedLinearModel)],
    val randomEffectType: String,
    val featureShardId: String)
  extends DatumScoringModel with RDDLike {

  override def sparkContext: SparkContext = modelsRDD.sparkContext

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!modelsRDD.getStorageLevel.isValid) modelsRDD.persist(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    if (modelsRDD.getStorageLevel.isValid) modelsRDD.unpersist()
    this
  }

  /**
   * NOTE: not used to reference models in the logic of our code, just used in logging for now!
   *
   * @param name The parent name for all RDDs in this class
   * @return This object with all its RDDs' name assigned
   */
  override def setName(name: String): this.type = {
    modelsRDD.setName(name)
    this
  }

  override def materialize(): this.type = {
    modelsRDD.count()
    this
  }

  /**
   * Compute the score for the dataset
   *
   * @param dataPoints The dataset
   * @return The score
   */
  override def score(dataPoints: RDD[(Long, GameDatum)]): KeyValueScore =
    RandomEffectModel.score(dataPoints, modelsRDD, randomEffectType, featureShardId)

  /**
   * Build a summary string for the model
   *
   * @return String representation
   */
  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(s"Random effect model of randomEffectType $randomEffectType, " +
        s"featureShardId $featureShardId summary:")
    stringBuilder.append(s"\nLength: ${modelsRDD.values.map(_.coefficients.means.length).stats()}")
    stringBuilder.append(s"\nMean: ${modelsRDD.values.map(_.coefficients.meansL2Norm).stats()}")
    if (modelsRDD.first()._2.coefficients.variancesOption.isDefined) {
      stringBuilder.append(s"\nvariance: ${modelsRDD.values.map(_.coefficients.variancesL2NormOption.get).stats()}")
    }
    stringBuilder.toString()
  }

  /**
   * Update the random effect model with new underlying models
   *
   * @param updatedModelsRdd The new underlying models, one per individual
   * @return The updated random effect model
   */
  def updateRandomEffectModel(updatedModelsRdd: RDD[(String, GeneralizedLinearModel)]): RandomEffectModel =
    new RandomEffectModel(updatedModelsRdd, randomEffectType, featureShardId)

  override def equals(that: Any): Boolean = {
    that match {
      case other: RandomEffectModel =>
        val sameMetaData =
          this.randomEffectType == other.randomEffectType && this.featureShardId == other.featureShardId
        lazy val sameCoefficientsRDD = this
          .modelsRDD
          .fullOuterJoin(other.modelsRDD)
          .mapPartitions { iterator =>
            Iterator.single(iterator.forall { case (_, (modelOpt1, modelOpt2)) =>
              modelOpt1.isDefined && modelOpt2.isDefined && modelOpt1.get.equals(modelOpt2.get)
            })
          }
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
   * @param modelsRDD The models to use for scoring
   * @param randomEffectType The random effect type
   * @param featureShardId The feature shard id
   * @return The scores
   */
  protected def score(
      dataPoints: RDD[(Long, GameDatum)],
      modelsRDD: RDD[(String, GeneralizedLinearModel)],
      randomEffectType: String,
      featureShardId: String): KeyValueScore = {

    val scores = dataPoints
      .map { case (uniqueId, gameData) =>
        val randomEffectId = gameData.idTypeToValueMap(randomEffectType)
        val features = gameData.featureShardContainer(featureShardId)
        (randomEffectId, (uniqueId, features))
      }
      .cogroup(modelsRDD)
      .flatMap { case (randomEffectId, (uniqueIdAndFeaturesIterable, modelsIterable)) =>
        // TODO(fastier): we should move that precondition upfront and check it only once, for speed
        assert(modelsIterable.size <= 1,
          s"More than one model (${modelsIterable.size}) found for individual Id $randomEffectId of " +
            s"random effect type $randomEffectType")

        if (modelsIterable.isEmpty) {
          uniqueIdAndFeaturesIterable.map { case (uniqueId, _) => (uniqueId, 0.0) }
        } else {
          val model = modelsIterable.head
          uniqueIdAndFeaturesIterable.map { case (uniqueId, features) =>
            (uniqueId, model.computeScore(features))
          }
        }
      }

    new KeyValueScore(scores)
  }
}
