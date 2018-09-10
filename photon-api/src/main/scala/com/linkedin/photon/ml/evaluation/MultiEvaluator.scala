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

import java.util.Objects

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.Types.UniqueSampleId

/**
 * Evaluator applied to a collection of samples grouped by some ID.
 *
 * Ex.  A [[MultiEvaluator]] X is created from a [[MultiEvaluatorType]] with [[MultiEvaluatorType.idTag]] "songId". The
 *      [[MultiEvaluator.ids]] of X will be (unique sample identifier, "songId" for that sample) pairs (e.g. (1, song1),
 *      (2, song2), (3, song1), ...). The unique sample identifier is used to join the "songId" to a
 *      (score, label, weight) tuple. The tuples are then grouped by "songId", creating a partial dataset for each ID
 *      (e.g. (song1, {...}), (song2, {...}), ...). The evaluation metric is computed for each songId using the partial
 *      dataset collected for it, and then averaged across all songIds. This average is the final evaluation metric
 *      returned by [[MultiEvaluator]] X.
 *
 * @param localEvaluator The underlying evaluator type
 * @param ids A [[RDD]] of (unique sample identifier, ID) pairs. The IDs are used to group samples, then the evaluation
 *            metric is computed on the groups per-ID and averaged. Such IDs can be thought of as a recommendation
 *            context (e.g. queryId when evaluating the relevance of search results for given queries).
 * @param labelAndOffsetAndWeights A [[RDD]] of (unique sample identifier, (label, offset, weight)) pairs
 */
abstract class MultiEvaluator(
  protected[ml] val localEvaluator: LocalEvaluator,
  protected[ml] val ids: RDD[(UniqueSampleId, String)],
  override protected[ml] val labelAndOffsetAndWeights: RDD[(UniqueSampleId, (Double, Double, Double))]) extends Evaluator {

  /**
   * Evaluate scores with labels and weights.
   *
   * @param scoresAndLabelsAndWeights A [[RDD]] of pairs (uniqueId, (score, label, weight))
   * @return Evaluation metric value
   */
  override protected[ml] def evaluateWithScoresAndLabelsAndWeights(
    scoresAndLabelsAndWeights: RDD[(UniqueSampleId, (Double, Double, Double))]): Double = {

    // Create a local copy of the localEvaluator, so that the underlying object won't get shipped to the executor nodes
    val localEvaluator = this.localEvaluator

    // TODO: Log which IDs had invalid evaluation metric values
    scoresAndLabelsAndWeights
      .join(ids)
      .map { case (_, (scoreLabelAndWeight, id)) => (id, scoreLabelAndWeight) }
      .groupByKey()
      .values
      .map(scoreLabelAndWeights => localEvaluator.evaluate(scoreLabelAndWeights.toArray))
      .filter(result => !(result.isInfinite || result.isNaN))
      .mean()
  }

  /**
   * Compares two [[MultiEvaluator]] objects.
   *
   * @param other Some other object
   * @return True if the both models conform to the equality contract and have the same model coefficients, false
   *         otherwise
   */
  override def equals(other: Any): Boolean = other match {
    case that: MultiEvaluator =>
      (this.localEvaluator == that.localEvaluator) && (this.ids == that.ids) && super.equals(that)

    case _ => false
  }

  /**
   * Returns a hash code value for the object.
   *
   * @return An [[Int]] hash code
   */
  override def hashCode: Int = Objects.hash(evaluatorType, labelAndOffsetAndWeights, localEvaluator, ids)
}
