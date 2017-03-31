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

import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.projector.ProjectionMatrixBroadcast
import com.linkedin.photon.ml.spark.BroadcastLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Representation of a factored random effect model.
 *
 * @param modelsInProjectedSpaceRDD The underlying model coefficients in projected space
 * @param projectionMatrixBroadcast The projector between the original and projected spaces
 * @param randomEffectType The random effect type
 * @param featureShardId The feature shard id
 */
protected[ml] class FactoredRandomEffectModel(
    override val modelsInProjectedSpaceRDD: RDD[(String, GeneralizedLinearModel)],
    val projectionMatrixBroadcast: ProjectionMatrixBroadcast,
    override val randomEffectType: String,
    override val featureShardId: String)
  extends RandomEffectModelInProjectedSpace(
    modelsInProjectedSpaceRDD,
    projectionMatrixBroadcast,
    randomEffectType,
    featureShardId) with BroadcastLike {

  /**
   * Update the factored random effect model with new models per individual.
   *
   * @param updatedModelsInProjectedSpaceRDD The new models with updated coefficients in projected space
   * @param updatedProjectionMatrixBroadcast The updated projection matrix
   * @return The updated factored random effect model in projected space
   */
  def updateFactoredRandomEffectModel(
    updatedModelsInProjectedSpaceRDD: RDD[(String, GeneralizedLinearModel)],
    updatedProjectionMatrixBroadcast: ProjectionMatrixBroadcast): FactoredRandomEffectModel = {

    val currType = this.modelType

    new FactoredRandomEffectModel(
        updatedModelsInProjectedSpaceRDD,
        updatedProjectionMatrixBroadcast,
        randomEffectType,
        featureShardId) {

      // TODO: The model types don't necessarily match, but checking each time is slow so copy the type for now
      override lazy val modelType: TaskType = currType
    }
  }

  /**
   * Build a summary string for the model.
   *
   * @return String representation
   */
  override def toSummaryString: String = {

    val stringBuilder = new StringBuilder(super.toSummaryString)
    stringBuilder.append("\nprojectionMatrix:")
    stringBuilder.append(s"\n${projectionMatrixBroadcast.projectionMatrix.toSummaryString}")

    stringBuilder.toString()
  }

  /**
   *
   * @return This object with all its broadcasted variables unpersisted
   */
  override def unpersistBroadcast(): this.type = {

    projectionMatrixBroadcast.unpersistBroadcast()

    this
  }
}
