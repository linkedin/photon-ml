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
package com.linkedin.photon.ml.evaluation

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

/**
  * Evaluator that computes area under the ROC curve
  *
  * @param labelAndOffsetAndWeights A [[RDD]] of (id, (label, offset, weight)) tuples
  * @param defaultScore The default score used to compute the metric
  */
protected[ml] class AreaUnderROCCurveEvaluator(
    labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))],
    defaultScore: Double = 0.0) extends Evaluator {

  protected val evaluatorType = AUC

  /**
    * Evaluate the scores of the model
    *
    * @param scores The scores to evaluate
    * @return Score metric value
    */
  override def evaluate(scores: RDD[(Long, Double)]): Double = {
    // Create a local copy of the defaultScore, so that the underlying object won't get shipped to the executor nodes
    val defaultScore = this.defaultScore
    val scoreAndLabels = scores
      .rightOuterJoin(labelAndOffsetAndWeights)
      .mapValues { case (scoreOption, (label, offset, _)) =>
        (scoreOption.getOrElse(defaultScore) + offset, label)
      }
      .values

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
