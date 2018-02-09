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

/**
 * Chain several validators together.
 */
class CompositeModelValidator[-GLM <: GeneralizedLinearModel : ClassTag](validators: ModelValidator[GLM]*)
  extends ModelValidator[GLM] {

  /**
   * Check that a model's predictions are valid according to multiple validators.
   *
   * @param model The GLM model to be validated
   * @param data The data used to validate the model
   */
  def validateModelPredictions(model: GLM, data: RDD[LabeledPoint]): Unit = {
    validators.foreach(v => { v.validateModelPredictions(model, data) })
  }
}
