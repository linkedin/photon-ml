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

import com.linkedin.photon.ml.Types.UniqueSampleId

/**
 * Batch version of the AUC evaluator.
 *
 * @param idTag Name of the column or metadata field containing IDs used to group samples (e.g. documentId or queryId)
 * @param ids A [[RDD]] of (unique sample identifier, ID) pairs. The IDs are used to group samples, then the evaluation
 *            metric is computed on the groups per-ID and averaged. Such IDs can be thought of as a recommendation
 *            context (e.g. queryId when evaluating the relevance of search results for given queries).
 * @param labelAndOffsetAndWeights A [[RDD]] of (unique sample identifier, (label, offset, weight)) pairs
 */
protected[ml] class AreaUnderROCCurveMultiEvaluator(
    idTag: String,
    override protected[ml] val ids: RDD[(UniqueSampleId, String)],
    override protected[ml] val labelAndOffsetAndWeights: RDD[(UniqueSampleId, (Double, Double, Double))])
  extends MultiEvaluator(AreaUnderROCCurveLocalEvaluator, ids, labelAndOffsetAndWeights) {

  val evaluatorType = MultiAUC(idTag)

  /**
   * Determine the better between two scores returned by this [[Evaluator]]. In some cases, the better score is higher
   * (e.g. AUC) and in others, the better score is lower (e.g. RMSE).
   *
   * @param score1 The first score to compare
   * @param score2 The second score to compare
   * @return True if the first score is better than the second
   */
  override def betterThan(score1: Double, score2: Double): Boolean = score1 > score2

  /**
   * Compares two [[AreaUnderROCCurveMultiEvaluator]] objects.
   *
   * @param other Some other object
   * @return True if the both models conform to the equality contract and have the same model coefficients, false
   *         otherwise
   */
  override def equals(other: Any): Boolean = other match {
    case that: AreaUnderROCCurveMultiEvaluator => super.equals(that)
    case _ => false
  }
}
