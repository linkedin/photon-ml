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
package com.linkedin.photon.ml.projector

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.{LabeledPoint, RandomEffectDataSet}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.spark.BroadcastLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Represents a broadcast projection matrix.
 *
 * @param projectionMatrixBroadcast The projection matrix
 */
protected[ml] class ProjectionMatrixBroadcast(projectionMatrixBroadcast: Broadcast[ProjectionMatrix])
  extends RandomEffectProjector
  with BroadcastLike
  with Serializable {

  val projectionMatrix: ProjectionMatrix = projectionMatrixBroadcast.value

  /**
   * Project the dataset from the original space to the projected space.
   *
   * @param randomEffectDataSet The input dataset in the original space
   * @return The same dataset in the projected space
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
   * Project a [[RDD]] of [[GeneralizedLinearModel]] [[Coefficients]] from the projected space back to the original
   * space.
   *
   * @param modelsRDD The input [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the projected space
   * @return The [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the original space
   */
  override def projectCoefficientsRDD(
      modelsRDD: RDD[(String, GeneralizedLinearModel)]): RDD[(String, GeneralizedLinearModel)] =
    modelsRDD.mapValues { model =>
      val oldCoefficients = model.coefficients
      model.updateCoefficients(
        Coefficients(
          projectionMatrixBroadcast.value.projectCoefficients(oldCoefficients.means),
          oldCoefficients.variancesOption))
    }

  /**
   * Project a [[RDD]] of [[GeneralizedLinearModel]] [[Coefficients]] from the original space to the projected space.
   *
   * @param modelsRDD The input [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the original space
   * @return The [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the projected space
   */
  override def transformCoefficientsRDD(
      modelsRDD: RDD[(String, GeneralizedLinearModel)]): RDD[(String, GeneralizedLinearModel)] =
    modelsRDD.mapValues { model =>
      val oldCoefficients = model.coefficients
      model.updateCoefficients(
        Coefficients(
          projectionMatrixBroadcast.value.projectFeatures(oldCoefficients.means),
          oldCoefficients.variancesOption))
    }

  /**
   * Project a [[NormalizationContext]] from the original space to the projected space.
   *
   * @param originalNormalizationContext The [[NormalizationContext]] in the original space
   * @return The same [[NormalizationContext]] in projected space
   */
  def projectNormalizationContext(originalNormalizationContext: NormalizationContext): NormalizationContext = {

    val factors = originalNormalizationContext.factorsOpt.map(factors => projectionMatrix.projectFeatures(factors))
    val shiftsAndIntercept = originalNormalizationContext
      .shiftsAndInterceptOpt
      .map { case (shifts, _) =>
        (projectionMatrix.projectFeatures(shifts), projectionMatrix.projectedInterceptId)
      }

    new NormalizationContext(factors, shiftsAndIntercept)
  }

  /**
   * Asynchronously delete cached copies of the [[ProjectionMatrix]] [[Broadcast]] on the executors.
   *
   * @return This [[ProjectionMatrixBroadcast]] with its [[ProjectionMatrix]] unpersisted
   */
  override def unpersistBroadcast(): this.type = {
    projectionMatrixBroadcast.unpersist()
    this
  }
}

object ProjectionMatrixBroadcast {

  /**
   * Generate random projection based broadcast projector
   *
   * @param randomEffectDataSet The input random effect dataset
   * @param projectedSpaceDimension The dimension of the projected feature space
   * @param isKeepingInterceptTerm Whether to keep the intercept in the original feature space
   * @param seed The seed of random number generator
   * @return The generated random projection based broadcast projector
   */
  protected[ml] def buildRandomProjectionBroadcastProjector(
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
