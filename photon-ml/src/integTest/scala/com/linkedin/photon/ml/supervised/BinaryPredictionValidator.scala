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
package com.linkedin.photon.ml.supervised

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Verify that on a particular data set, the model only produces finite predictions
 *
 * TODO LOW: think about adding support for other thresholds.
 */
class BinaryPredictionValidator[-GLM <: GeneralizedLinearModel with BinaryClassifier: ClassTag]
    extends ModelValidator[GLM] {

  override def validateModelPredictions(model:GLM, data:RDD[LabeledPoint]) : Unit = {
    val predictions = model.predictClassAllWithThreshold(data.map(x => x.features),
      MathConst.POSITIVE_RESPONSE_THRESHOLD)
    val invalidCount = predictions.filter(x => x != BinaryClassifier.negativeClassLabel &&
        x != BinaryClassifier.positiveClassLabel
    ).count()
    if (invalidCount > 0) {
      throw new IllegalStateException(s"Found [$invalidCount] samples with invalid predictions (expect " +
          s"[$BinaryClassifier.negativeClassLabel] or [$BinaryClassifier.positiveClassLabel]")
    }
  }
}
