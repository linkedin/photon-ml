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
 * Batch version of the AUROC evaluator.
 *
 * @param idTag Name of the column or metadata field containing IDs used to group samples (e.g. documentId or queryId)
 * @param ids A [[RDD]] of (unique sample ID, tag ID value) pairs. The IDs are used to group samples, then the
 *            evaluation metric is computed on the groups per-ID and averaged. Such IDs can be thought of as a
 *            recommendation context (e.g. queryId when evaluating the relevance of search results for given queries).
 */
protected[ml] class AreaUnderROCCurveMultiEvaluator(
    protected val idTag: String,
    override protected val ids: RDD[(UniqueSampleId, String)])
  extends MultiEvaluator(AreaUnderROCCurveLocalEvaluator, ids) {

  val evaluatorType = MultiAUC(idTag)

  /**
   * Compares two [[AreaUnderROCCurveMultiEvaluator]] objects.
   *
   * @param other Some other object
   * @return True if the both models conform to the equality contract and have the same model coefficients, false
   *         otherwise
   */
  override def equals(other: Any): Boolean = other match {
    case that: AreaUnderROCCurveMultiEvaluator => this.idTag == that.idTag && super.equals(that)
    case _ => false
  }

  /**
   * Returns a hash code value for the object.
   *
   * @return An [[Int]] hash code
   */
  override def hashCode: Int = Objects.hash(evaluatorType, localEvaluator, ids, idTag)
}
