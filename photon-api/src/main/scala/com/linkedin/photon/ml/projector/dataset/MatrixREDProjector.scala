/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.projector.dataset

import org.apache.spark.broadcast.Broadcast

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.{LabeledPoint, RandomEffectDataset}
import com.linkedin.photon.ml.projector.vector.MatrixProjector
import com.linkedin.photon.ml.spark.BroadcastLike

/**
 * Project each per-entity dataset stored in a [[RandomEffectDataset]] between spaces using a projection matrix.
 *
 * @param matrixProjectorBroadcast A shared [[MatrixProjector]] for projecting models between spaces
 */
protected[ml] class MatrixREDProjector(matrixProjectorBroadcast: Broadcast[MatrixProjector])
  extends RandomEffectDatasetProjector with BroadcastLike with Serializable {

  /**
   * Project a [[RandomEffectDataset]] from the original space to the projected space.
   *
   * @param randomEffectDataset The [[RandomEffectDataset]] in the original space
   * @return The same [[RandomEffectDataset]] in the projected space
   */
  override def projectForward(randomEffectDataset: RandomEffectDataset): RandomEffectDataset = {

    val activeData = randomEffectDataset.activeData
    val passiveData = randomEffectDataset.passiveData

    val projectedActiveData = activeData.mapValues(_.projectForward(matrixProjectorBroadcast.value))
    val projectedPassiveData = passiveData.mapValues { case (shardId, LabeledPoint(response, features, offset, weight)) =>
      (shardId, LabeledPoint(response, matrixProjectorBroadcast.value.projectForward(features), offset, weight))
    }

    randomEffectDataset.update(projectedActiveData, projectedPassiveData)
  }

  /**
   * Project a [[RandomEffectDataset]] from the projected space to the original space.
   *
   * @param randomEffectDataset The [[RandomEffectDataset]] in the projected space
   * @return The same [[RandomEffectDataset]] in the original space
   */
  override def projectBackward(randomEffectDataset: RandomEffectDataset): RandomEffectDataset = {

    val activeData = randomEffectDataset.activeData
    val passiveData = randomEffectDataset.passiveData

    val projectedActiveData = activeData.mapValues(_.projectBackward(matrixProjectorBroadcast.value))
    val projectedPassiveData = passiveData.mapValues { case (shardId, LabeledPoint(response, features, offset, weight)) =>
      (shardId, LabeledPoint(response, matrixProjectorBroadcast.value.projectBackward(features), offset, weight))
    }

    randomEffectDataset.update(projectedActiveData, projectedPassiveData)
  }

  /**
   * Asynchronously delete cached copies of the [[Broadcast]] [[MatrixProjector]] on the executors.
   *
   * @return This [[MatrixREDProjector]] with [[matrixProjectorBroadcast]] marked for deletion
   */
  override def unpersistBroadcast(): MatrixREDProjector = {

    matrixProjectorBroadcast.unpersist()

    this
  }
}

object MatrixREDProjector {

  /**
   * Generate a [[MatrixREDProjector]] based on Gaussian random projection matrices.
   *
   * @param randomEffectDataset The input random effect dataset
   * @param projectedSpaceDimension The dimension of the projected feature space
   * @param isKeepingInterceptTerm Whether to keep the intercept in the original feature space
   * @param seed The seed of random number generator
   * @return The generated random projection based broadcast projector
   */
  protected[ml] def build(
      randomEffectDataset: RandomEffectDataset,
      projectedSpaceDimension: Int,
      isKeepingInterceptTerm: Boolean,
      seed: Long = MathConst.RANDOM_SEED): MatrixREDProjector = {

    val sparkContext = randomEffectDataset.sparkContext
    val originalSpaceDimension = randomEffectDataset.activeData.first()._2.numFeatures
    val randomProjectionMatrix = MatrixProjector.buildGaussianRandomMatrixProjector(
      originalSpaceDimension,
      projectedSpaceDimension,
      isKeepingInterceptTerm,
      seed)
    val randomProjectionMatrixBroadcast = sparkContext.broadcast[MatrixProjector](randomProjectionMatrix)

    new MatrixREDProjector(randomProjectionMatrixBroadcast)
  }
}
