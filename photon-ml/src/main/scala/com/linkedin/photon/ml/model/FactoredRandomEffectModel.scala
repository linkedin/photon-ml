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

import com.linkedin.photon.ml.BroadcastLike
import com.linkedin.photon.ml.projector.ProjectionMatrixBroadcast
import org.apache.spark.rdd.RDD

/**
  * Representation of a factored random effect model
  *
  * @param coefficientsRDDInProjectedSpace The underlying model coefficients in projected space
  * @param projectionMatrixBroadcast The projector between the original and projected spaces
  * @param randomEffectId The random effect type id
  * @param featureShardId The feature shard id
  */
protected[ml] class FactoredRandomEffectModel(
    override val coefficientsRDDInProjectedSpace: RDD[(String, Coefficients)],
    val projectionMatrixBroadcast: ProjectionMatrixBroadcast,
    override val randomEffectId: String,
    override val featureShardId: String)
  extends RandomEffectModelInProjectedSpace(
    coefficientsRDDInProjectedSpace,
    projectionMatrixBroadcast,
    randomEffectId,
    featureShardId) with BroadcastLike {

  override def unpersistBroadcast(): this.type = {
    projectionMatrixBroadcast.unpersistBroadcast()
    this
  }

  /**
    * Build a summary string for the model
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
    * Update the factored random effect model with new models per individual
    *
    * @param updatedCoefficientsRDDInProjectedSpace The new models with updated coefficients in projected space
    * @param updatedProjectionMatrixBroadcast The updated projection matrix
    * @return The updated factored random effect model in projected space
    */
  def updateFactoredRandomEffectModel(
      updatedCoefficientsRDDInProjectedSpace: RDD[(String, Coefficients)],
      updatedProjectionMatrixBroadcast: ProjectionMatrixBroadcast): FactoredRandomEffectModel = {

    new FactoredRandomEffectModel(
      updatedCoefficientsRDDInProjectedSpace,
      updatedProjectionMatrixBroadcast,
      randomEffectId,
      featureShardId)
  }
}
