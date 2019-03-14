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

import java.lang.{Double => JDouble}
import java.util.{Arrays => JArrays, Comparator => JComparator}

import com.linkedin.photon.ml.constants.MathConst._

/**
 * Local evaluator for Area Under ROC curve.
 */
protected[evaluation] object AreaUnderROCCurveLocalEvaluator extends LocalEvaluator {

  /**
   * Compute the AUROC for the given data.
   *
   * @param scoresAndLabelsAndWeights An [[Array]] of (score, label, weight) tuples
   * @return The AUROC
   */
  override def evaluate(scoresAndLabelsAndWeights: Array[(Double, Double, Double)]): Double = {

    //Directly calling scoresAndLabelAndWeight.sort would cause some performance issue with large arrays
    val comparator = new JComparator[(Double, Double, Double)]() {
      override def compare(tuple1: (Double, Double, Double), tuple2: (Double, Double, Double)): Int = {
        JDouble.compare(tuple1._1, tuple2._1)
      }
    }
    JArrays.sort(scoresAndLabelsAndWeights, comparator.reversed())

    var rawAUC = 0.0
    var totalPositiveCount = 0.0
    var totalNegativeCount = 0.0
    var currentPositiveCount = 0.0
    var currentNegativeCount = 0.0
    var previousScore = scoresAndLabelsAndWeights.head._1

    scoresAndLabelsAndWeights.foreach { case (score, label, weight) =>
      if (score != previousScore) {
        rawAUC += totalPositiveCount * currentNegativeCount + currentPositiveCount * currentNegativeCount / 2.0
        totalPositiveCount += currentPositiveCount
        totalNegativeCount += currentNegativeCount
        currentPositiveCount = 0.0
        currentNegativeCount = 0.0
      }
      if (label > POSITIVE_RESPONSE_THRESHOLD) {
        currentPositiveCount += weight
      } else {
        currentNegativeCount += weight
      }
      previousScore = score
    }
    rawAUC += totalPositiveCount * currentNegativeCount + currentPositiveCount * currentNegativeCount / 2.0
    totalPositiveCount += currentPositiveCount
    totalNegativeCount += currentNegativeCount

    //normalize AUC over the total area
    rawAUC / (totalPositiveCount * totalNegativeCount)
  }
}
