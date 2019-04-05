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
package com.linkedin.photon.ml.algorithm

import com.linkedin.photon.ml.data.RandomEffectDataset
import com.linkedin.photon.ml.model.{Coefficients, RandomEffectModel}

/**
 * Trait to encapsulate [[RandomEffectModel]] projection. Needed as the random effects have their feature space
 * collapsed to reduce the amount of memory used and training time.
 */
trait ModelProjection extends Coordinate[RandomEffectDataset] {

  /**
   * Project a [[RandomEffectModel]] from the original space to the projected space.
   *
   * @param randomEffectModel The [[RandomEffectModel]] in the original space
   * @return The same [[RandomEffectModel]] in the projected space
   */
  protected[algorithm] def projectModelForward(randomEffectModel: RandomEffectModel): RandomEffectModel = {

    // Left join the models to projectors for cases where we have a prior model but no new model (and hence no
    // projectors)
    val linearSubspaceProjectorsRDD = dataset.projectors
    val newModels = randomEffectModel
      .modelsRDD
      .leftOuterJoin(linearSubspaceProjectorsRDD)
      .mapValues { case (model, projectorOpt) =>
        projectorOpt
          .map { projector =>
            val oldCoefficients = model.coefficients
            val newCoefficients = Coefficients(
              projector.projectForward(oldCoefficients.means),
              oldCoefficients.variancesOption.map(projector.projectForward))

            model.updateCoefficients(newCoefficients)
          }
          .getOrElse(model)
      }

    randomEffectModel.update(newModels)
  }

  /**
   * Project a [[RandomEffectModel]] from the projected space to the original space.
   *
   * @param randomEffectModel The [[RandomEffectModel]] in the projected space
   * @return The same [[RandomEffectModel]] in the original space
   */
  protected[algorithm] def projectModelBackward(randomEffectModel: RandomEffectModel): RandomEffectModel = {

    // Left join the models to projectors for cases where we have a prior model but no new model (and hence no
    // projectors)
    val linearSubspaceProjectorsRDD = dataset.projectors
    val newModels = randomEffectModel
      .modelsRDD
      .leftOuterJoin(linearSubspaceProjectorsRDD)
      .mapValues { case (model, projectorOpt) =>
        projectorOpt
          .map { projector =>
            val oldCoefficients = model.coefficients
            val newCoefficients = Coefficients(
              projector.projectBackward(oldCoefficients.means),
              oldCoefficients.variancesOption.map(projector.projectBackward))

            model.updateCoefficients(newCoefficients)
          }
          .getOrElse(model)
      }

    randomEffectModel.update(newModels)
  }
}
