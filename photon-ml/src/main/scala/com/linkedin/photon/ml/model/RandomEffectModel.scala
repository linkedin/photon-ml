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


import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD._
import org.apache.spark.storage.StorageLevel
import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.data.{KeyValueScore, GameData}

/**
 * Representation of a random effect model
 *
 * @param coefficientsRDD the coefficients
 * @param randomEffectId the random effect type id
 * @param featureShardId the feature shard id
 * @author xazhang
 */
class RandomEffectModel(
    val coefficientsRDD: RDD[(String, Coefficients)],
    val randomEffectId: String,
    val featureShardId: String)
  extends Model with RDDLike {

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
   * @param dataPoints the dataset
   * @return the score
   */
  override def score(dataPoints: RDD[(Long, GameData)]): KeyValueScore = {
    RandomEffectModel.score(dataPoints, coefficientsRDD, randomEffectId, featureShardId)
  }

  /**
   * Build a summary string for the model
   *
   * @return string representation
   */
  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(s"Random effect model of randomEffectId $randomEffectId, " +
        s"featureShardId $featureShardId summary:")
    stringBuilder.append(s"\nLength: ${coefficientsRDD.values.map(_.means.length).stats()}")
    stringBuilder.append(s"\nMean: ${coefficientsRDD.map(_._2.meansL2Norm).stats()}")
    if (coefficientsRDD.first()._2.variancesOption.isDefined) {
      stringBuilder.append(s"\nvariance: ${coefficientsRDD.map(_._2.variancesL2NormOption.get).stats()}")
    }
    stringBuilder.toString()
  }

  /**
   * Update the random effect model
   *
   * @param updatedCoefficientsRDD the coefficients
   * @return the updated model
   */
  def updateRandomEffectModel(updatedCoefficientsRDD: RDD[(String, Coefficients)]): RandomEffectModel = {
    new RandomEffectModel(updatedCoefficientsRDD, randomEffectId, featureShardId)
  }
}

object RandomEffectModel {

  /**
   * Compute the score for the dataset
   *
   * @param dataPoints the dataset
   * @param randomEffectId the random effect type id
   * @param featureShardId the feature shard id
   * @return the score
   */
  protected def score(
      dataPoints: RDD[(Long, GameData)],
      coefficientsRDD: RDD[(String, Coefficients)],
      randomEffectId: String,
      featureShardId: String): KeyValueScore = {

    val scores = dataPoints.map { case (globalId, gameData) =>
      val individualId = gameData.randomEffectIdToIndividualIdMap(randomEffectId)
      val features = gameData.featureShardContainer(featureShardId)
      (individualId, (globalId, features))
    }.cogroup(coefficientsRDD).flatMap { case (individualId, (globalIdAndFeaturesIterable, coefficientsIterable)) =>
      assert(coefficientsIterable.size <= 1, s"More than one coefficients (${coefficientsIterable.size}) " +
          s"found for individual Id $individualId of random effect Id $randomEffectId")
      if (coefficientsIterable.isEmpty) {
        globalIdAndFeaturesIterable.map { case (globalId, _) => (globalId, 0.0) }
      } else {
        val coefficients = coefficientsIterable.head
        globalIdAndFeaturesIterable.map { case (globalId, features) => (globalId, coefficients.computeScore(features)) }
      }
    }
    new KeyValueScore(scores)
  }
}
