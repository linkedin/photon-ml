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

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.projector.RandomEffectProjector

/**
 * Representation of a random effect model in projected space
 *
 * @param coefficientsRDDInProjectedSpace the coefficients in projected space
 * @param randomEffectProjector random effect projector
 * @param randomEffectId the random effect type id
 * @param featureShardId the feature shard id
 * @author xazhang
 */
class RandomEffectModelInProjectedSpace(
    val coefficientsRDDInProjectedSpace: RDD[(String, Coefficients)],
    val randomEffectProjector: RandomEffectProjector,
    override val randomEffectId: String,
    override val featureShardId: String)
    extends RandomEffectModel(randomEffectProjector.projectCoefficientsRDD(coefficientsRDDInProjectedSpace),
      randomEffectId, featureShardId) {

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!coefficientsRDDInProjectedSpace.getStorageLevel.isValid) coefficientsRDDInProjectedSpace.persist(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    if (coefficientsRDDInProjectedSpace.getStorageLevel.isValid) coefficientsRDDInProjectedSpace.unpersist()
    this
  }

  override def setName(name: String): this.type = {
    coefficientsRDDInProjectedSpace.setName(name)
    this
  }

  override def materialize(): this.type = {
    coefficientsRDDInProjectedSpace.count()
    this
  }

  /**
   * Build a summary string for the model
   *
   * @return string representation
   */
  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(s"Random effect model with projector with randomEffectId $randomEffectId, " +
        s"featureShardId $featureShardId summary:")
    stringBuilder.append("\ncoefficientsRDDInProjectedSpace:")
    stringBuilder.append(s"\nLength: ${coefficientsRDDInProjectedSpace.values.map(_.means.length).stats()}")
    stringBuilder.append(s"\nMean: ${coefficientsRDDInProjectedSpace.map(_._2.meansL2Norm).stats()}")
    if (coefficientsRDDInProjectedSpace.first()._2.variancesOption.isDefined) {
      stringBuilder.append(s"\nVar: ${coefficientsRDDInProjectedSpace.map(_._2.variancesL2NormOption.get).stats()}")
    }
    stringBuilder.toString()
  }

  /**
   * Convert the projected space model into a random effect model
   *
   * @return the random effect model
   */
  protected[ml] def toRandomEffectModel: RandomEffectModel = {
    new RandomEffectModel(coefficientsRDDInProjectedSpace, randomEffectId, featureShardId)
  }

  /**
   * Update the random effect model in projected space
   *
   * @param updatedCoefficientsRDDInProjectedSpace the coefficients in projected space
   * @return the updated model
   */
  def updateRandomEffectModelInProjectedSpace(
      updatedCoefficientsRDDInProjectedSpace: RDD[(String, Coefficients)]): RandomEffectModelInProjectedSpace = {

    new RandomEffectModelInProjectedSpace(updatedCoefficientsRDDInProjectedSpace, randomEffectProjector, randomEffectId,
      featureShardId)
  }
}
