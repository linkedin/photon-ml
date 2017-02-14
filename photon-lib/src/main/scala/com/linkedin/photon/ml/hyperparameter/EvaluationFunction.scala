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
package com.linkedin.photon.ml.hyperparameter

import breeze.linalg.DenseVector

/**
 * Base trait for all Evaluation functions
 *
 * An evaluation function is the integration point between the hyperparameter tuning module and an estimator, or any
 * system that can unpack a vector of values and produce a real evaluation.
 */
trait EvaluationFunction[T] {

  /**
   * Performs the evaluation
   *
   * @param hyperParameters the vector of hyperparameter values under which to evaluate the function
   * @return a tuple of the evaluated value and the original output from the inner estimator
   */
  def apply(hyperParameters: DenseVector[Double]): (Double, T)

  /**
   * Extracts a vector representation from the hyperparameters associated with the original estimator output
   *
   * @param result the original estimator output
   * @return vector representation
   */
  def vectorizeParams(result: T): DenseVector[Double]

  /**
   * Extracts the evaluated value from the original estimator output
   *
   * @param result the original estimator output
   * @return the evaluated value
   */
  def getEvaluationValue(result: T): Double
}
