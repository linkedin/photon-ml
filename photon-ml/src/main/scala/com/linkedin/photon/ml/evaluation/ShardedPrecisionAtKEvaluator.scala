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


/**
 * Sharded version of the precision@k evaluator
 *
 * @param k The cut-off rank based on which the precision is computed (precision @ k)
 * @param idType Type of the id, e.g., documentId or queryId.
 * @param ids Ids based on which the labels and scores are grouped (sharded) to compute the evaluation metric for each
 *            shard/group. Such ids can be thought as a recommendation context, e.g. in evaluating the relevance of
 *            search results of given a query, the id can be the query itself.
 * @param labelAndOffsetAndWeights a [[RDD]] of (id, (labels, offsets, weights)) pairs
 * Interface for evaluation implementations at the [[RDD]] level
 */
protected[ml] class ShardedPrecisionAtKEvaluator(
    k: Int,
    idType: String,
    override protected[ml] val ids: RDD[(Long, String)],
    override protected[ml] val labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))])
  extends ShardedEvaluator(new PrecisionAtKLocalEvaluator(k), ids, labelAndOffsetAndWeights) {

  protected[ml]  val evaluatorType = ShardedPrecisionAtK(k, idType)

  override def betterThan(score1: Double, score2: Double): Boolean = score1 > score2
}
