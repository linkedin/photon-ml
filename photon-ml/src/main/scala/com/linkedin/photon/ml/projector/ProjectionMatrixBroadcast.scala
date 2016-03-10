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
package com.linkedin.photon.ml.projector

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.BroadcastLike
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.{RandomEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.model.Coefficients

/**
 * Represents a broadcast projection matrix
 *
 * @param projectionMatrixBroadcast the projection matrix
 * @author xazhang
 */
class ProjectionMatrixBroadcast(projectionMatrixBroadcast: Broadcast[ProjectionMatrix])
    extends RandomEffectProjector with BroadcastLike with Serializable {

  val projectionMatrix = projectionMatrixBroadcast.value

  /**
   * Project the sharded data set from the original space to the projected space
   *
   * @param randomEffectDataSet The input sharded data set in the original space
   * @return The sharded data set in the projected space
   */
  override def projectRandomEffectDataSet(randomEffectDataSet: RandomEffectDataSet): RandomEffectDataSet = {
    val activeData = randomEffectDataSet.activeData
    val passiveDataOption = randomEffectDataSet.passiveDataOption
    val projectedActiveData = activeData.mapValues(_.projectFeatures(projectionMatrixBroadcast.value))

    val projectedPassiveData = if (passiveDataOption.isDefined) {
      passiveDataOption.map(_.mapValues { case (shardId, LabeledPoint(response, features, offset, weight)) =>
        val projectedFeatures = projectionMatrixBroadcast.value.projectFeatures(features)
        (shardId, LabeledPoint(response, projectedFeatures, offset, weight))
      })
    } else {
      None
    }

    randomEffectDataSet.update(projectedActiveData, projectedPassiveData)
  }

  /**
   * Project a [[RDD]] of [[Coefficients]] from the projected space back to the original space
   *
   * @param coefficientsRDD The input [[RDD]] of [[Coefficients]] in the projected space
   * @return The [[RDD]] of [[Coefficients]] in the original space
   */
  override def projectCoefficientsRDD(coefficientsRDD: RDD[(String, Coefficients)]): RDD[(String, Coefficients)] = {
    coefficientsRDD.mapValues { case Coefficients(mean, varianceOption) =>
      Coefficients(projectionMatrixBroadcast.value.projectCoefficients(mean), varianceOption)
    }
  }

  override def unpersistBroadcast(): this.type = {
    projectionMatrixBroadcast.unpersist()
    this
  }
}

object ProjectionMatrixBroadcast {

  /**
   * Generate random projection based broadcast projector
   *
   * @param randomEffectDataSet The input random effect data set
   * @param projectedSpaceDimension The dimension of the projected feature space
   * @param isKeepingInterceptTerm Whether to keep the intercept in the original feature space
   * @param seed The seed of random number generator
   * @return The generated random projection based broadcast projector
   */
  def buildRandomProjectionBroadcastProjector(
      randomEffectDataSet: RandomEffectDataSet,
      projectedSpaceDimension: Int,
      isKeepingInterceptTerm: Boolean,
      seed: Long = MathConst.RANDOM_SEED): ProjectionMatrixBroadcast = {

    val sparkContext = randomEffectDataSet.sparkContext
    val originalSpaceDimension = randomEffectDataSet.activeData.first()._2.numFeatures
    val randomProjectionMatrix = ProjectionMatrix.buildGaussianRandomProjectionMatrix(projectedSpaceDimension,
      originalSpaceDimension, isKeepingInterceptTerm, seed)
    val randomProjectionMatrixBroadcast = sparkContext.broadcast[ProjectionMatrix](randomProjectionMatrix)

    new ProjectionMatrixBroadcast(randomProjectionMatrixBroadcast)
  }
}
