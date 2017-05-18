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

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

/**
 * Evaluator that computes area under the ROC curve.
 *
 * @param labelAndOffsetAndWeights A [[RDD]] of (id, (label, offset, weight)) tuples
 */
protected[ml] class AreaUnderROCCurveEvaluator(
    override protected[ml] val labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))]) extends Evaluator {

  val evaluatorType = EvaluatorType.AUC

  /**
   * Evaluate scores with labels and weights.
   *
   * @param scoresAndLabelsAndWeights A [[RDD]] of pairs (uniqueId, (score, label, weight)).
   * @return Evaluation metric value
   */
  override protected[ml] def evaluateWithScoresAndLabelsAndWeights(
    scoresAndLabelsAndWeights: RDD[(Long, (Double, Double, Double))]): Double = {

    val scoreAndLabels = scoresAndLabelsAndWeights.map { case (uniqueId, (score, label, weight)) => (score, label) }
    new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
  }

  /**
   * Determine the best between two scores returned by the evaluator. In some cases, the better score is higher
   * (e.g. AUC) and in others, the better score is lower (e.g. RMSE).
   *
   * @param score1 The first score to compare
   * @param score2 The second score to compare
   * @return True if the first score is better than the second
   */
  override def betterThan(score1: Double, score2: Double): Boolean = score1 > score2
}
