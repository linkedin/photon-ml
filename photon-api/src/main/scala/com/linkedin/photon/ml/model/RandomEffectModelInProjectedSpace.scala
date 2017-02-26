/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
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

import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.projector.RandomEffectProjector
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Representation of a random effect model in projected space.
 *
 * @param modelsInProjectedSpaceRDD The underlying models with coefficients in projected space
 * @param randomEffectProjector The projector between the original and projected spaces
 * @param randomEffectType The random effect type
 * @param featureShardId The feature shard id
 */
protected[ml] class RandomEffectModelInProjectedSpace(
    val modelsInProjectedSpaceRDD: RDD[(String, GeneralizedLinearModel)],
    val randomEffectProjector: RandomEffectProjector,
    override val randomEffectType: String,
    override val featureShardId: String)
  extends RandomEffectModel(
    randomEffectProjector.projectCoefficientsRDD(modelsInProjectedSpaceRDD),
    randomEffectType,
    featureShardId) {

  /**
   *
   * @param storageLevel The storage level
   * @return This object with all its RDDs' storage level set
   */
  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!modelsInProjectedSpaceRDD.getStorageLevel.isValid) modelsInProjectedSpaceRDD.persist(storageLevel)
    this
  }

  /**
   *
   * @return This object with all its RDDs unpersisted
   */
  override def unpersistRDD(): this.type = {
    if (modelsInProjectedSpaceRDD.getStorageLevel.isValid) modelsInProjectedSpaceRDD.unpersist()
    this
  }

  /**
   *
   * @param name The parent name for the model RDD in this class
   * @return This object with all its RDDs' name assigned
   */
  override def setName(name: String): this.type = {
    modelsInProjectedSpaceRDD.setName(name)
    this
  }

  /**
   *
   * @return This object with all its RDDs materialized
   */
  override def materialize(): this.type = {
    modelsInProjectedSpaceRDD.count()
    this
  }

  /**
   * Summarize this model in text format.
   *
   * @return A model summary in text format.
   */
  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(s"Random effect model with projector with " +
      s"randomEffectType $randomEffectType, featureShardId $featureShardId summary:")
    stringBuilder.append("\ncoefficientsRDDInProjectedSpace:")
    stringBuilder.append(s"\nLength: ${modelsInProjectedSpaceRDD.values.map(_.coefficients.means.length).stats()}")
    stringBuilder.append(s"\nMean: ${modelsInProjectedSpaceRDD.values.map(_.coefficients.meansL2Norm).stats()}")
    if (modelsInProjectedSpaceRDD.first()._2.coefficients.variancesOption.isDefined) {
      stringBuilder.append(
        s"\nVar: ${modelsInProjectedSpaceRDD.values.map(_.coefficients.variancesL2NormOption.get).stats()}")
    }
    stringBuilder.toString()
  }

  /**
   * Convert the projected space model into a [[RandomEffectModel]].
   *
   * @return A [[RandomEffectModel]]
   */
  def toRandomEffectModel: RandomEffectModel = {
    val currType = this.modelType
    new RandomEffectModel(modelsInProjectedSpaceRDD, randomEffectType, featureShardId) {
      // TODO: The model types don't necessarily match, but checking each time is slow so copy the type for now
      override lazy val modelType: TaskType = currType
    }
  }

  /**
   * Update the random effect model in projected space with new sub-models (one per random effect ID).
   *
   * @param updatedModelsRDDInProjectedSpace The new sub-models with coefficients in projected space, one per random
   *                                         effect ID
   * @return The updated random effect model in projected space
   */
  def updateRandomEffectModelInProjectedSpace(
      updatedModelsRDDInProjectedSpace: RDD[(String, GeneralizedLinearModel)]): RandomEffectModelInProjectedSpace = {
    val currType = this.modelType
    new RandomEffectModelInProjectedSpace(
      updatedModelsRDDInProjectedSpace,
      randomEffectProjector,
      randomEffectType,
      featureShardId) {
      // TODO: The model types don't necessarily match, but checking each time is slow so copy the type for now
      override lazy val modelType: TaskType = currType
    }
  }
}
