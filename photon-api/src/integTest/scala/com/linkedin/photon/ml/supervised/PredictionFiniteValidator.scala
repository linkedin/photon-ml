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

import breeze.linalg.Vector
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.Regression

/**
  * Verify that on a particular data set, the model only produces finite predictions.
  */
class PredictionFiniteValidator extends ModelValidator[GeneralizedLinearModel] {

  /**
   *
   * @param model The GLM model to be validated
   * @param data The data used to validate the model
   */
  override def validateModelPredictions(model: GeneralizedLinearModel, data: RDD[LabeledPoint]) : Unit = {
    val features: RDD[Vector[Double]] = data.map(_.features)
    val predictions: RDD[Double] = model match {
      case r: Regression => r.predictAll(features)
      case b: BinaryClassifier => b.predictClassAll(features, 0.5)
      case _ =>
        throw new IllegalArgumentException("Don't know how to handle models of type [" + model.getClass.getName + "]")
    }
    val invalidCount: Long = predictions.filter(!java.lang.Double.isFinite(_)).count()

    if (invalidCount > 0) {
      throw new IllegalStateException("Found [" + invalidCount + "] samples with invalid (NaN or +/-Inf) predictions")
    }
  }
}
