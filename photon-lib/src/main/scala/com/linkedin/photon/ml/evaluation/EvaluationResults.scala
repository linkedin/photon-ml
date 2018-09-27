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
package com.linkedin.photon.ml.evaluation

/**
 * Helper object which wraps around a [[Map]] of [[EvaluatorType]] to evaluation metric, and tracks the primary
 * [[EvaluatorType]] / evaluation metric (e.g. used for model selection).
 *
 * @param evaluations A [[Map]] of [[EvaluatorType]] to evaluation metric
 * @param primaryEvaluator The primary [[EvaluatorType]]
 */
case class EvaluationResults(evaluations: Map[EvaluatorType, Double], primaryEvaluator: EvaluatorType) {

  require(evaluations.contains(primaryEvaluator), "Primary evaluator not found in evaluations")

  /**
   * Return the evaluation metric for the primary [[Evaluator]].
   *
   * @return The primary evaluation metric
   */
  def primaryEvaluation: Double = evaluations(primaryEvaluator)
}

