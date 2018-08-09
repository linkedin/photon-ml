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
package com.linkedin.photon.ml.hyperparameter.estimators

import breeze.linalg.DenseVector

/**
 * Base trait for prediction transformations. A prediction transformation is applied to a model's predictions before
 * they're returned or integrated over.
 */
trait PredictionTransformation {

  def isMaxOpt: Boolean

  /**
   * Applies the transformation. Implementing classes should provide specific transformations here.
   *
   * @param predictiveMeans predictive mean output from the model
   * @param predictiveVariances predictive variance output from the model
   * @return the transformed predictions
   */
  def apply(
      predictiveMeans: DenseVector[Double],
      predictiveVariances: DenseVector[Double]): DenseVector[Double]
}
