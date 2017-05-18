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

/**
 * Evaluator sharded with the specified ids.
 *
 * @param localEvaluator The underlying evaluator type
 * @param ids IDs based on which the labels and scores are grouped (sharded) to compute the evaluation metric for each
 *            shard/group. Such ids can be thought as a recommendation context, e.g. in evaluating the relevance of
 *            search results of given a query, the id can be the query itself.
 * @param labelAndOffsetAndWeights A [[RDD]] of (id, (labels, offsets, weights)) pairs
 */
abstract class ShardedEvaluator(
  protected[ml] val localEvaluator: LocalEvaluator,
  protected[ml] val ids: RDD[(Long, String)],
  protected[ml] val labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))]) extends Evaluator {

  /**
   * Evaluate scores with labels and weights.
   *
   * @param scoresAndLabelsAndWeights A [[RDD]] of pairs (uniqueId, (score, label, weight))
   * @return Evaluation metric value
   */
  override protected[ml] def evaluateWithScoresAndLabelsAndWeights(
    scoresAndLabelsAndWeights: RDD[(Long, (Double, Double, Double))]): Double = {

    // Create a local copy of the localEvaluator, so that the underlying object won't get shipped to the executor nodes
    val localEvaluator = this.localEvaluator
    scoresAndLabelsAndWeights
      .join(ids)
      .map { case (uniqueId, (scoreLabelAndWeight, id)) => (id, scoreLabelAndWeight) }
      .groupByKey()
      .values
      .map(scoreLabelAndWeights => localEvaluator.evaluate(scoreLabelAndWeights.toArray))
      .mean()
  }
}
