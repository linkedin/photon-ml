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

import com.linkedin.photon.ml.function.PointwiseSquareLossFunction

/**
 * Evaluator for squared loss
 *
 * @param labelAndOffsetAndWeights a [[RDD]] of (id, (labels, offsets, weights)) pairs
 */
protected[ml] class SquaredLossEvaluator(
    override protected[ml] val labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))]) extends Evaluator {

  protected[ml] val evaluatorType = SquaredLoss

  override protected[ml] def evaluateWithScoresAndLabelsAndWeights(
    scoresAndLabelsAndWeights: RDD[(Long, (Double, Double, Double))]): Double = {

    scoresAndLabelsAndWeights.map { case (_, (score, label, weight)) =>
        weight * PointwiseSquareLossFunction.loss(score, label)._1
      }
      .reduce(_ + _)
  }

  /**
   * Determine the best between two scores returned by the evaluator. In some cases, the better score is higher
   * (e.g. AUC) and in others, the better score is lower (e.g. RMSE).
   *
   * @param score1 the first score to compare
   * @param score2 the second score to compare
   * @return true if the first score is better than the second
   */
  override def betterThan(score1: Double, score2: Double): Boolean = score1 < score2
}
