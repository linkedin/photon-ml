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

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.RandomEffectDataSet
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * A trait that performs two types of projections:
 * <ul>
 * <li>
 *   Project the random effect data set from the original space to the projected space, usually as a pre-processing
 *   step before the model training phase.
 * </li>
 * <li>
 *   Project the model coefficients from the projected space back to the original space after training the model,
 *   before scoring a data set in the original space.
 * </li>
 * </ul>
 */
protected[ml] trait RandomEffectProjector {

  /**
   * Project the data set from the original space to the projected space.
   *
   * @param randomEffectDataSet The input data set in the original space
   * @return The same data set in the projected space
   */
  def projectRandomEffectDataSet(randomEffectDataSet: RandomEffectDataSet): RandomEffectDataSet

  /**
   * Project a [[RDD]] of [[GeneralizedLinearModel]] [[Coefficients]] from the projected space back to the original
   * space.
   *
   * @param modelsRDD The input [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the projected space
   * @return The [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the original space
   */
  def projectCoefficientsRDD(modelsRDD: RDD[(String, GeneralizedLinearModel)]): RDD[(String, GeneralizedLinearModel)]

  /**
   * Project a [[RDD]] of [[GeneralizedLinearModel]] [[Coefficients]] from the original space to the projected space.
   *
   * @param modelsRDD The input [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the original space
   * @return The [[RDD]] of [[GeneralizedLinearModel]] with [[Coefficients]] in the projected space
   */
  def transformCoefficientsRDD(modelsRDD: RDD[(String, GeneralizedLinearModel)]): RDD[(String, GeneralizedLinearModel)]

}

object RandomEffectProjector {

  /**
   * Builds a random effect projector instance.
   *
   * @param randomEffectDataSet The data set to project
   * @param projectorType The type of the projector
   * @return A new [[RandomEffectProjector]]
   */
  protected[ml] def build(
      randomEffectDataSet: RandomEffectDataSet,
      projectorType: ProjectorType): RandomEffectProjector = projectorType match {

    case RandomProjection(projectedSpaceDimension) =>
      ProjectionMatrixBroadcast.buildRandomProjectionBroadcastProjector(
        randomEffectDataSet, projectedSpaceDimension, isKeepingInterceptTerm = true)

    case IdentityProjection =>
      new IdentityProjector

    case IndexMapProjection =>
      IndexMapProjectorRDD.buildIndexMapProjector(randomEffectDataSet)

    case _ =>
      throw new UnsupportedOperationException(
        s"Projector type $projectorType for random effect data set is not supported")
  }
}
