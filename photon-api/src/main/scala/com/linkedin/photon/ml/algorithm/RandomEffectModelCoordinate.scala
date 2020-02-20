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

import org.apache.spark.sql.DataFrame

import com.linkedin.photon.ml.Types.{FeatureShardId, REType}
import com.linkedin.photon.ml.data.InputColumnsNames
import com.linkedin.photon.ml.model.{DatumScoringModel, RandomEffectModel}

/**
 * The optimization problem coordinate for a pre-trained random effect model.
 *
 * @param dataset The training dataset
 */
class RandomEffectModelCoordinate(
  rEType: REType,
  dataset: DataFrame,
  featureShardId: FeatureShardId,
  inputColumnsNames: InputColumnsNames)
  extends ModelCoordinate {

  /**
   * Score the effect-specific dataset in the coordinate with the input model.
   *
   * @param model The input model
   * @return The output scores
   */
  override protected def updateOffset(model: DatumScoringModel) = {

    model match {
      case randomEffectModel: RandomEffectModel =>
        RandomEffectCoordinate.updateOffset(dataset, randomEffectModel, featureShardId, rEType, inputColumnsNames)

      case _ =>
        throw new UnsupportedOperationException(
          s"Updating scores with model of type ${model.getClass} in ${this.getClass} is not supported")
    }
  }
}
