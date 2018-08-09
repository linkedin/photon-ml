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
package com.linkedin.photon.ml.hyperparameter.criteria

import breeze.linalg.DenseVector
import breeze.numerics.sqrt

import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.hyperparameter.estimators.PredictionTransformation

/**
 * Confidence bound selection criterion. This transformation produces either an upper or lower confidence bound,
 * depending on how the evaluator compares values.
 *
 * @param evaluator the evaluator
 * @param explorationFactor a factor that determines the tradeoff between exploration and exploitation during search:
 *   higher values favor exploration.
 */
class ConfidenceBound(
    evaluator: Evaluator,
    explorationFactor: Double = 2.0)
  extends PredictionTransformation {

  def isMaxOpt: Boolean = evaluator.betterThan(1.0, 0.0)

  /**
   * Applies the confidence bound transformation to the model output
   *
   * @param predictiveMeans predictive mean output from the model
   * @param predictiveVariances predictive variance output from the model
   * @return the confidence bounds
   */
  def apply(
      predictiveMeans: DenseVector[Double],
      predictiveVariances: DenseVector[Double]): DenseVector[Double] = {

    // PBO Eq. 3
    val confidenceBounds = explorationFactor * sqrt(predictiveVariances)
    val upper = predictiveMeans + confidenceBounds
    val lower = predictiveMeans - confidenceBounds


    if (evaluator.betterThan(upper(0), lower(0))) {
      upper
    } else {
      lower
    }
  }
}
