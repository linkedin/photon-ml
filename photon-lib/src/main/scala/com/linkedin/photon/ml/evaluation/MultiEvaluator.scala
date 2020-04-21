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
 * [[Evaluator]] to compute an evaluation metric per a collection of samples grouped by some ID.
 *
 * Ex.  A [[MultiEvaluator]] X is created from a [[MultiEvaluatorType]] with [[MultiEvaluatorType.idTag]] "songId". The
 *      [[MultiEvaluator.ids]] of X will be (unique sample identifier, "songId" for that sample) pairs (e.g. (1, song1),
 *      (2, song2), (3, song1), ...). The unique sample identifier is used to join the "songId" to a
 *      (score, label, weight) tuple. The tuples are then grouped by "songId", creating a partial dataset for each ID
 *      (e.g. (song1, {...}), (song2, {...}), ...). The evaluation metric is computed for each songId using the partial
 *      dataset collected for it, and then averaged across all songIds. This average is the final evaluation metric
 *      returned by [[MultiEvaluator]] X.
 *
 * @param localEvaluator A [[LocalEvaluator]] used to compute evaluation metrics per group of samples
 */
abstract class MultiEvaluator(
    protected val localEvaluator: LocalEvaluator,
    protected val ids: RDD[(UniqueSampleId, String)])
  extends Evaluator {

  type ScoredData = (UniqueSampleId, (Double, Double, Double))

  /**
   * Compute an evaluation metric on a per-group basis and average the results.
   *
   * @param scoresAndLabelsAndWeights A [[RDD]] of pairs (uniqueId, (score, label, weight))
   * @return Evaluation metric value
   */
  override def evaluate(
      scoresAndLabelsAndWeights: RDD[(UniqueSampleId, (Double, Double, Double))]): Double = {

    // Create a local copy of the localEvaluator, so that the underlying object won't get shipped to the executor nodes
    val localEvaluator = this.localEvaluator

    // EvaluationSuite guarantees that all validation data is in scoresAndLabelsAndWeights RDD, and ids RDD is directly
    // mapped from validation data. Thus, inner join should be guaranteed to not lose any data.
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
   * Compute an evaluation metric on a per-group basis and save the per group results.
   *
   * @param scoresAndLabelsAndWeights A [[RDD]] of pairs (uniqueId, (score, label, weight))
   * @return Evaluation metric value
   */
  def evaluatePerGroup(
      scoresAndLabelsAndWeights: RDD[(UniqueSampleId, (Double, Double, Double))]):
    (Double, Option[RDD[(String, Double)]]) = {

    // Create a local copy of the localEvaluator, so that the underlying object won't get shipped to the executor nodes
    val localEvaluator = this.localEvaluator

    // EvaluationSuite guarantees that all validation data is in scoresAndLabelsAndWeights RDD, and ids RDD is directly
    // mapped from validation data. Thus, inner join should be guaranteed to not lose any data.
    val groupedData = scoresAndLabelsAndWeights
      .join(ids)
      .map { case (_, (scoreLabelAndWeight, id)) => (id, scoreLabelAndWeight) }
      .groupByKey()

      val perGroupEvaluation = groupedData
        .mapValues(scoreLabelAndWeights => localEvaluator.evaluate(scoreLabelAndWeights.toArray))
        .filter(results => !(results._2.isInfinite || results._2.isNaN))

      val meanEvaluation = perGroupEvaluation
        .values
        .mean()

      (meanEvaluation, Some(perGroupEvaluation))
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
      (this.localEvaluator == that.localEvaluator) && this.ids.eq(that.ids) && super.equals(that)

    case _ =>
      false
  }

  /**
   * Returns a hash code value for the object.
   *
   * @return An [[Int]] hash code
   */
  override def hashCode: Int = Objects.hash(evaluatorType, localEvaluator, ids)
}
