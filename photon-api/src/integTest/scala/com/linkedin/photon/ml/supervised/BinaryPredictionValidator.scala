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
package com.linkedin.photon.ml.supervised

import scala.reflect.ClassTag

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Verify that on a particular data set, the model only produces finite predictions.
 */
class BinaryPredictionValidator[-GLM <: GeneralizedLinearModel with BinaryClassifier: ClassTag]
  extends ModelValidator[GLM] {

  // TODO: Think about adding support for other thresholds.
  /**
   * Check that all predictions match one of the two binary classification labels.
   *
   * @param model The GLM model to be validated
   * @param data The data used to validate the model
   */
  override def validateModelPredictions(model: GLM, data: RDD[LabeledPoint]): Unit = {
    val predictions = model.predictClassAll(
      data.map(x => x.features),
      MathConst.POSITIVE_RESPONSE_THRESHOLD)
    val invalidCount = predictions
      .filter { x =>
        (x != BinaryClassifier.negativeClassLabel) && (x != BinaryClassifier.positiveClassLabel)
      }
      .count()

    if (invalidCount > 0) {
      throw new IllegalStateException(s"Found [$invalidCount] samples with invalid predictions (expect " +
          s"[$BinaryClassifier.negativeClassLabel] or [$BinaryClassifier.positiveClassLabel]")
    }
  }
}
