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

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.Regression

/**
 * Verify that on a particular data set, the model only produces non-negative predictions.
 */
class NonNegativePredictionValidator[-GLM <: GeneralizedLinearModel with Regression: ClassTag]
  extends ModelValidator[GLM] {

  /**
   * Check that each prediction is non-negative.
   *
   * @param model The GLM model to be validated
   * @param data The data used to validate the model
   */
  override def validateModelPredictions(model:GLM, data:RDD[LabeledPoint]): Unit = {

    val predictions = model.predictAll(data.map(x => x.features))
    val invalidCount = predictions.filter(x => x < 0).count()

    if (invalidCount > 0) {
      throw new IllegalStateException(s"Found [$invalidCount] samples with invalid predictions (expect " +
          s"non-negative labels only).")
    }
  }
}
