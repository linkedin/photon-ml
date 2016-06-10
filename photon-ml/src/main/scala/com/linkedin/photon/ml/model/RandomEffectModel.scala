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
  * @param randomEffectId The random effect type id
  * @param featureShardId The feature shard id
  */
protected[ml] class RandomEffectModel(
    val modelsRDD: RDD[(String, GeneralizedLinearModel)],
    val randomEffectId: String,
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
    RandomEffectModel.score(dataPoints, modelsRDD, randomEffectId, featureShardId)

  /**
    * Build a summary string for the model
    *
    * @return String representation
    */
  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(s"Random effect model of randomEffectId $randomEffectId, " +
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
    new RandomEffectModel(updatedModelsRdd, randomEffectId, featureShardId)

  override def equals(that: Any): Boolean = {
    that match {
      case other: RandomEffectModel =>
        val sameMetaData = this.randomEffectId == other.randomEffectId && this.featureShardId == other.featureShardId
        val sameCoefficientsRDD = this
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
    * @param coefficientsRDD The models to use for scoring
    * @param randomEffectId The random effect type id
    * @param featureShardId The feature shard id
    * @return The scores
    */
  protected def score(
      dataPoints: RDD[(Long, GameDatum)],
      modelsRDD: RDD[(String, GeneralizedLinearModel)],
      randomEffectId: String,
      featureShardId: String): KeyValueScore = {

    val scores = dataPoints
      .map { case (globalId, gameData) =>
        val individualId = gameData.randomEffectIdToIndividualIdMap(randomEffectId)
        val features = gameData.featureShardContainer(featureShardId)
        (individualId, (globalId, features))
      }
      .cogroup(modelsRDD)
      .flatMap { case (individualId, (globalIdAndFeaturesIterable, modelsIterable)) =>
        assert(modelsIterable.size <= 1,
          s"More than one model (${modelsIterable.size}) found for individual Id $individualId of " +
            s"random effect Id $randomEffectId")

        if (modelsIterable.isEmpty) {
          globalIdAndFeaturesIterable.map { case (globalId, _) => (globalId, 0.0) }
        } else {
          val model = modelsIterable.head
          globalIdAndFeaturesIterable.map { case (globalId, features) =>
            (globalId, model.computeScore(features))
          }
        }
      }

    new KeyValueScore(scores)
  }
}
