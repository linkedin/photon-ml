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
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.model.{DatumScoringModel, RandomEffectModel}

/**
 * The optimization problem coordinate for a pre-trained random effect model.
 *
 * @param dataSet The training dataset
 */
class RandomEffectModelCoordinate(dataSet: RandomEffectDataset) extends ModelCoordinate(dataSet) {

  /**
   * Score the effect-specific dataset in the coordinate with the input model.
   *
   * @param model The input model
   * @return The output scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = {
    model match {
      case randomEffectModel: RandomEffectModel =>
        RandomEffectCoordinate.score(dataSet, randomEffectModel)

      case _ =>
        throw new UnsupportedOperationException(
          s"Updating scores with model of type ${model.getClass} in ${this.getClass} is not supported")
    }
  }
}
