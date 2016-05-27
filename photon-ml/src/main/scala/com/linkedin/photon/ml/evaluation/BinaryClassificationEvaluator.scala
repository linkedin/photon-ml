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
  * Evaluator for binary classification problems
  *
  * @param labelAndOffsets a [[RDD]] of (id, (label, offset)) pairs
  * @param defaultScore the default score used to compute the metric
  * @author xazhang
  */
protected[ml] class BinaryClassificationEvaluator(
    labelAndOffsets: RDD[(Long, (Double, Double))],
    defaultScore: Double = 0.0) extends Evaluator {

  /**
    * Evaluate the scores of the model
    *
    * @param scores the scores to evaluate
    * @return score metric value
    */
  override def evaluate(scores: RDD[(Long, Double)]): Double = {
    // Create a local copy of the defaultScore, so that the underlying object won't get shipped to the executor nodes
    val defaultScore = this.defaultScore
    val scoreAndLabels = scores.rightOuterJoin(labelAndOffsets).mapValues { case (scoreOption, (label, offset)) =>
      (scoreOption.getOrElse(defaultScore) + offset, label)
    }.values

    new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
  }

  /**
    * Determine the best between two scores returned by the evaluator. In some cases, the better score is higher
    * (e.g. AUC) and in others, the better score is lower (e.g. RMSE).
    *
    * @param score1 the first score to compare
    * @param score2 the second score to compare
    * @return true if the first score is better than the second
    */
  override def betterThan(score1: Double, score2: Double): Boolean = score1 > score2
}
