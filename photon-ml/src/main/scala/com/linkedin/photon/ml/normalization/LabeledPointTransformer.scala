/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.normalization

import breeze.linalg.Vector
import com.linkedin.photon.ml.data
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * A scaler to scale [[http://www.scalanlp.org/api/breeze/#breeze.linalg.Vector breeze.linalg.Vector]] in the
 * [[data.LabeledPoint LabeledPoint]].
 * @author dpeng
 */
@SerialVersionUID(1L)
class LabeledPointTransformer(vectorTransformer: Transformer[Vector[Double]]) extends Transformer[LabeledPoint] {
  /**
   * Transform/Scale a [[data.LabeledPoint LabeledPoint]]. It only transforms the underlying
   * vector in the [[data.LabeledPoint LabeledPoint]]. The sparsity of the vector is
   * preserved.
   * @param input Input [[data.LabeledPoint LabeledPoint]]
   * @return Output [[data.LabeledPoint LabeledPoint]]
   */
  override def transform(input: LabeledPoint): LabeledPoint = {
    val LabeledPoint(label, features, weight, offset) = input
    val transformedFeatures = vectorTransformer.transform(features)
    LabeledPoint(label, transformedFeatures, weight, offset)
  }
}
