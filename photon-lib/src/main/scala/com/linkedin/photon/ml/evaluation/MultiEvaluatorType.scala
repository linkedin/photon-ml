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

import scala.util.matching.Regex

import com.linkedin.photon.ml.util.MathUtils

/**
 * Trait for an evaluator applied to a collection of samples grouped by some ID.
 */
trait MultiEvaluatorType extends EvaluatorType {

  /**
   * Name of the column or metadata field containing IDs used to group records for evaluation (e.g. documentId, queryId,
   * etc.)
   */
  val idTag: String
}

object MultiEvaluatorType {

  val shardedEvaluatorIdNameSplitter = ":"

  /**
   * Get the set of unique ID tags for a group of [[MultiEvaluatorType]]s.
   *
   * @param evaluators A list of [[MultiEvaluatorType]]
   * @return The set of unique ID column names used by the evaluators
   */
  def getMultiEvaluatorIdTags(evaluators: Seq[EvaluatorType]): Set[String] =
    evaluators
      .flatMap {
        case shardedEvaluatorType: MultiEvaluatorType => Some(shardedEvaluatorType.idTag)
        case _ => None
      }
      .toSet
}

case class MultiPrecisionAtK(k: Int, override val idTag: String) extends MultiEvaluatorType {

  require(k > 0, s"Position k must be greater than 0: $k")

  val name = s"PRECISION@$k${MultiEvaluatorType.shardedEvaluatorIdNameSplitter}$idTag"
  val op = MathUtils.greaterThan _
}

object MultiPrecisionAtK {

  val batchPrecisionAtKPattern: Regex =
    s"(?i:PRECISION)@(\\d+)${MultiEvaluatorType.shardedEvaluatorIdNameSplitter}(.*)".r
}

case class MultiAUC(override val idTag: String) extends MultiEvaluatorType {

  val name = s"AUC${MultiEvaluatorType.shardedEvaluatorIdNameSplitter}$idTag"
  val op = MathUtils.greaterThan _
}

object MultiAUC {

  val batchAUCPattern: Regex = s"(?i:AUC)${MultiEvaluatorType.shardedEvaluatorIdNameSplitter}(.*)".r
}
