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

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.util.Utils

/**
 * Evaluator for squared loss
 *
 * @param labelAndOffsetAndWeights label and offset weights
 * @param defaultScore default score
 * @author xazhang
 */
class SquaredLossEvaluator(
    labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))],
    defaultScore: Double = 0.0) extends Evaluator {

  /**
   * Evaluate scores
   *
   * @param score the scores to evaluate
   * @return score metric value
   */
  override def evaluate(scores: RDD[(Long, Double)]): Double = {
    val defaultScore = this.defaultScore

    val scoreAndLabelAndWeights = scores.rightOuterJoin(labelAndOffsetAndWeights)
        .mapValues { case (scoreOption, (label, offset, weight)) =>
      (scoreOption.getOrElse(defaultScore) + offset, (label, weight))
    }.values

    scoreAndLabelAndWeights.map { case (score, (label, weight)) =>
      val diff = score - label
      weight * diff * diff
    }.reduce(_ + _)
  }
}
