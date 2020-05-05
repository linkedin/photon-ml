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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.spark.RDDLike

/**
 * A [[Set]] of [[Evaluator]] objects and the labels + offsets + weights used to evaluate scored data.
 *
 * @param evaluators The set of [[Evaluator]] objects to use to compute validation metrics
 * @param primaryEvaluator The [[Evaluator]] which computes the most important validation metric (e.g. the metric used
 *                         for model selection)
 * @param labelAndOffsetAndWeights A [[RDD]] of (unique ID, (label, offset, weight) tuple), which is joined with a
 *                                 [[RDD]] of scores to compute validation metrics
 * @param savePerGroupEvaluation Whether to save per group evaluation in a separate output file
 */
class EvaluationSuite(
    val evaluators: Set[Evaluator],
    val primaryEvaluator: Evaluator,
    labelAndOffsetAndWeights: RDD[(UniqueSampleId, (Double, Double, Double))],
    savePerGroupEvaluation: Boolean = false)
  extends RDDLike {

  checkInvariants()

  /**
   * Test the conditions that should always hold true for any [[EvaluationSuite]].
   */
  private def checkInvariants(): Unit = {

    require(evaluators.nonEmpty, "Evaluator set cannot be empty")
    require(evaluators.contains(primaryEvaluator), "Primary evaluator is not present in the set of evaluators")
  }

  /**
   * Evaluate each metric for the given scores.
   *
   * @param scores The scores to evaluate
   * @return The evaluation metric values as [[EvaluationResults]]
   */
  protected[ml] def evaluate(scores: RDD[(UniqueSampleId, Double)]): EvaluationResults = {

    // Possible for all models to be missing a score for some datum, meaning the score for a datum is missing even after
    // summing scores from all models. Thus, need a leftOuterJoin.
    val scoresAndLabelsAndWeights = labelAndOffsetAndWeights
      .leftOuterJoin(scores)
      .mapValues { case ((label, offset, weight), scoreOpt) =>
        (scoreOpt.getOrElse(MathConst.DEFAULT_SCORE) + offset, label, weight)
      }
      .persist()

    val evaluations = evaluators
      .map {
        case evaluator: SingleEvaluator =>
          (evaluator.evaluatorType, (evaluator.evaluate(scoresAndLabelsAndWeights.values), None))

        case multiEvaluator: MultiEvaluator =>
          if (savePerGroupEvaluation) {
            (multiEvaluator.evaluatorType, multiEvaluator.evaluatePerGroup(scoresAndLabelsAndWeights))
          } else {
            (multiEvaluator.evaluatorType, (multiEvaluator.evaluate(scoresAndLabelsAndWeights), None))
          }

        case otherEvaluator =>
          throw new IllegalArgumentException(s"Cannot process Evaluator of type '${otherEvaluator.getClass}'")
      }
      .toMap

    scoresAndLabelsAndWeights.unpersist()

    EvaluationResults(evaluations, primaryEvaluator.evaluatorType)
  }

  //
  // RDDLike functions
  //

  /**
   * Get the Spark context used by the [[RDD]] of this [[EvaluationSuite]].
   *
   * @return The Spark context
   */
  protected[ml] def sparkContext: SparkContext = labelAndOffsetAndWeights.sparkContext

  /**
   * Assign a given name to the [[labelAndOffsetAndWeights]] RDD.
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   * @param name The name for the [[labelAndOffsetAndWeights]] RDD
   * @return This [[EvaluationSuite]] with the name of the [[labelAndOffsetAndWeights]] RDD assigned
   */
  protected[ml] def setName(name: String): EvaluationSuite = {

    labelAndOffsetAndWeights.setName(s"$name")

    this
  }

  /**
   * Set the storage level of the [[labelAndOffsetAndWeights]] RDD, and persist their values across the cluster the
   * first time they are computed.
   *
   * @param storageLevel The storage level
   * @return This [[EvaluationSuite]] with the storage level of all of the [[labelAndOffsetAndWeights]] RDD set
   */
  protected[ml] def persistRDD(storageLevel: StorageLevel): EvaluationSuite = {

    if (!labelAndOffsetAndWeights.getStorageLevel.isValid) {
      labelAndOffsetAndWeights.persist(storageLevel)
    }

    this
  }

  /**
   * Mark the [[labelAndOffsetAndWeights]] RDD as non-persistent, and remove all blocks for it from memory and disk.
   *
   * @return This [[EvaluationSuite]] with the [[labelAndOffsetAndWeights]] RDD marked as non-persistent
   */
  protected[ml] def unpersistRDD(): EvaluationSuite = {

    if (labelAndOffsetAndWeights.getStorageLevel.isValid) {
      labelAndOffsetAndWeights.unpersist()
    }

    this
  }

  /**
   * Materialize all the [[labelAndOffsetAndWeights]] RDD (Spark [[RDD]]s are lazy evaluated: this method forces them to
   * be evaluated).
   *
   * @return This [[EvaluationSuite]] with the [[labelAndOffsetAndWeights]] RDD materialized
   */
  protected[ml] def materialize(): EvaluationSuite = {

    labelAndOffsetAndWeights.count()

    this
  }
}

object EvaluationSuite {

  /**
   * Helper function to construct an [[EvaluationSuite]] from an ordered set of [[Evaluator]] objects (passed as a [[Seq]]).
   *
   * @param evaluators Ordered set of [[Evaluator]] objects
   * @param labelAndOffsetAndWeights [[RDD]] of samples: their unique ID, label, offset, and weight
   * @param savePerGroupEvaluation Whether to save per group evaluation in a separate output file
   * @return A new [[EvaluationSuite]]
   */
  def apply(
      evaluators: Seq[Evaluator],
      labelAndOffsetAndWeights: RDD[(UniqueSampleId, (Double, Double, Double))],
      savePerGroupEvaluation: Boolean = false): EvaluationSuite = {

    val numEvaluators = evaluators.size
    val numUniqueEvaluators = evaluators.map(_.getEvaluatorName).toSet.size

    require(numEvaluators == numUniqueEvaluators, "Evaluators contain duplicates")

    // Use first Evaluator as the primary Evaluator
    new EvaluationSuite(evaluators.toSet, evaluators.head, labelAndOffsetAndWeights, savePerGroupEvaluation)
  }
}
