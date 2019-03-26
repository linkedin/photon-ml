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

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.function.glm.SquaredLossFunction

/**
 * Evaluator for squared loss.
 */
object SquaredLossEvaluator extends SingleEvaluator {

  val evaluatorType = EvaluatorType.SquaredLoss

  /**
   * Compute squared loss for the given data.
   *
   * @param scoresAndLabelsAndWeights A [[RDD]] of scored data
   * @return The squared loss
   */
  override def evaluate(scoresAndLabelsAndWeights: RDD[(Double, Double, Double)]): Double =
    scoresAndLabelsAndWeights
      .map { case (score, label, weight) =>
        // Squared loss function is calculated as 1/2 (z - y)^2 for mathematical convenience.
        // However, RMSE should be using (z - y)^2 for calculation, hence the multiplying by 2 here:
        2 * weight * SquaredLossFunction.lossAndDzLoss(score, label)._1
      }
      .reduce(_ + _)
}
