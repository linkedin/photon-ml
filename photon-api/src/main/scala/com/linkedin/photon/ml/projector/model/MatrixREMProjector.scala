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
package com.linkedin.photon.ml.projector.model

import org.apache.spark.broadcast.Broadcast

import com.linkedin.photon.ml.model.{Coefficients, RandomEffectModel}
import com.linkedin.photon.ml.projector.vector.MatrixProjector
import com.linkedin.photon.ml.spark.BroadcastLike

/**
 * Project each component model of a [[RandomEffectModel]] between spaces using a projection matrix.
 *
 * @param matrixProjectorBroadcast A shared [[MatrixProjector]] for projecting models between spaces
 */
class MatrixREMProjector(matrixProjectorBroadcast: Broadcast[MatrixProjector])
  extends RandomEffectModelProjector with BroadcastLike {

  /**
   * Project a [[RandomEffectModel]] from the original space to the projected space.
   *
   * @param randomEffectModel The [[RandomEffectModel]] in the original space
   * @return The same [[RandomEffectModel]] in the projected space
   */
  override def projectForward(randomEffectModel: RandomEffectModel): RandomEffectModel = {

    val newModels = randomEffectModel
      .modelsRDD
      .mapValues { model =>
        val oldCoefficients = model.coefficients
        val newCoefficients = Coefficients(
          matrixProjectorBroadcast.value.projectForward(oldCoefficients.means),
          oldCoefficients.variancesOption.map(matrixProjectorBroadcast.value.projectForward))

        model.updateCoefficients(newCoefficients)
      }

    randomEffectModel.update(newModels)
  }

  /**
   * Project a [[RandomEffectModel]] from the projected space to the original space.
   *
   * @param randomEffectModel The [[RandomEffectModel]] in the projected space
   * @return The same [[RandomEffectModel]] in the original space
   */
  override def projectBackward(randomEffectModel: RandomEffectModel): RandomEffectModel = {

    val newModels = randomEffectModel
      .modelsRDD
      .mapValues { model =>
        val oldCoefficients = model.coefficients
        val newCoefficients = Coefficients(
          matrixProjectorBroadcast.value.projectBackward(oldCoefficients.means),
          oldCoefficients.variancesOption.map(matrixProjectorBroadcast.value.projectBackward))

        model.updateCoefficients(newCoefficients)
      }

    randomEffectModel.update(newModels)
  }

  /**
   * Asynchronously delete cached copies of the [[Broadcast]] [[MatrixProjector]] on the executors.
   *
   * @return This [[MatrixREMProjector]] with [[matrixProjectorBroadcast]] marked for deletion
   */
  override def unpersistBroadcast(): MatrixREMProjector = {

    matrixProjectorBroadcast.unpersist()

    this
  }
}
