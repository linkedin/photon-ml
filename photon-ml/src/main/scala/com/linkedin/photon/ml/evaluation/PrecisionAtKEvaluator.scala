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

/**
 * Evaluator for Precision@k. Definition of this evaluator along with terminologies used here (e.g., documentId) can be
 * found at [[https://en.wikipedia.org/wiki/Information_retrieval#Precision_at_K]]. No special tiebreaker is used if
 * there are ties in scores, and one of them will be picked randomly.
 *
 * @param k The cut-off rank based on which the precision is computed (precision @ k)
 * @param labelAndOffsetAndWeights A [[RDD]] of (id, (label, offset, weight)) tuples
 * @param documentIds Document ids based on which the labels and scores are grouped to form documents in order to
 *                    compute precision @ K. Such document ids can be thought as a recommendation context, e.g. in
 *                    evaluating the relevance of search results of given a query - one would use the query id as a
 *                    documentId.
 * @param documentIdName Name of the document Id, e.g., documentId or queryId.
 * @param defaultScore The default score used to compute the metric
 */
class PrecisionAtKEvaluator(
    k: Int,
    labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))],
    documentIds: RDD[(Long, String)],
    documentIdName: String,
    defaultScore: Double = 0.0) extends Evaluator {

  protected val evaluatorType = PrecisionAtK(k, documentIdName)

  override def evaluate(scores: RDD[(Long, Double)]): Double = {
    // Create a local copy of the defaultScore, so that the underlying object won't get shipped to the executor nodes
    val k = this.k
    val defaultScore = this.defaultScore
    val scoreAndLabels = scores
      .rightOuterJoin(labelAndOffsetAndWeights)
      .mapValues { case (scoreOption, (label, offset, _)) =>
        (scoreOption.getOrElse(defaultScore) + offset, label)
      }
    val posThreshold = MathConst.POSITIVE_RESPONSE_THRESHOLD
    documentIds
      .join(scoreAndLabels)
      .values
      .groupByKey()
      .values
      .map(_.toArray.sortBy(_._1)(Ordering[Double].reverse).take(k).count(_._2 > posThreshold) * 1.0 / k)
      .mean()
  }

  override def betterThan(score1: Double, score2: Double): Boolean = score1 > score2
}
