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
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.Regression
import org.apache.spark.rdd.RDD


class PredictionNonNegativeValidator extends ModelValidator[GeneralizedLinearModel] {

  override def validateModelPredictions(model:GeneralizedLinearModel, data:RDD[LabeledPoint]) : Unit = {
    val features = data.map { x => x.features }
    var predictions:RDD[Double] = null
    model match {
      case r:Regression =>
        predictions = r.predictAll(features)

      case b:BinaryClassifier =>
        predictions = b.predictClassAllWithThreshold(features, 0.5)

      case _ =>
        throw new IllegalArgumentException("Don't know how to handle models of type [" + model.getClass.getName + "]")
    }

    val invalidCount = predictions.filter(x => x<0).count()
    if (invalidCount > 0) {
      throw new IllegalStateException("Found [" + invalidCount + "] samples with invalid negative predictions")
    }
  }
}
