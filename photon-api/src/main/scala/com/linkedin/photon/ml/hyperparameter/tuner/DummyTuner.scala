/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.hyperparameter.tuner

import breeze.linalg.DenseVector

import com.linkedin.photon.ml.HyperparameterTuningMode.HyperparameterTuningMode
import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction

/**
 * A dummy hyper-parameter tuner which runs an empty operation.
 */
class DummyTuner[T] extends HyperparameterTuner[T] {

  /**
   * Search hyper-parameters to optimize the model
   *
   * @param n The number of points to find
   * @param dimension Numbers of hyper-parameters to be tuned
   * @param mode Hyper-parameter tuning mode (random or Bayesian)
   * @param evaluationFunction Function that evaluates points in the space to real values
   * @param evaluator the original evaluator
   * @param observations Observations made prior to searching, from this data set (not mean-centered)
   * @param priorObservations Observations made prior to searching, from past data sets (mean-centered)
   * @param discreteParams Map that specifies the indices of discrete parameters and their numbers of discrete values
   * @return A Seq of the found results
   */
  def search(
      n: Int,
      dimension: Int,
      mode: HyperparameterTuningMode,
      evaluationFunction: EvaluationFunction[T],
      evaluator: Evaluator,
      observations: Seq[(DenseVector[Double], Double)],
      priorObservations: Seq[(DenseVector[Double], Double)] = Seq(),
      discreteParams: Map[Int, Int] = Map()): Seq[T] = Seq()
}
