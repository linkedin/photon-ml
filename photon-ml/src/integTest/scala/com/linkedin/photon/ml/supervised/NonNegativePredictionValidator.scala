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

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.Regression
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Verify that on a particular data set, the model only produces nonnegative predictions
 *
 * @author asaha
 */
class NonNegativePredictionValidator[-GLM <: GeneralizedLinearModel with Regression: ClassTag]
    extends ModelValidator[GLM] {

  override def validateModelPredictions(model:GLM, data:RDD[LabeledPoint]) : Unit = {
    val predictions = model.predictAll(data.map(x => x.features))
    val invalidCount = predictions.filter(x => x < 0).count()
    if (invalidCount > 0) {
      throw new IllegalStateException(s"Found [$invalidCount] samples with invalid predictions (expect " +
          s"non-negative labels only).")
    }
  }
}
