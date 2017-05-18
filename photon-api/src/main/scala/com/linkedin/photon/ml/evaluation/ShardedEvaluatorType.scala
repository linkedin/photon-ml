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

/**
 * Trait for sharded evaluator (i.e. evaluator applied to a data set sharded by ID).
 */
trait ShardedEvaluatorType extends EvaluatorType {
  /**
   * ID column used to shard the data for evaluation (e.g. documentId, queryId, etc.)
   */
  val idColumn: String
}

object ShardedEvaluatorType {
  val shardedEvaluatorIdNameSplitter = ":"

  /**
   * Get all id types used to compute sharded evaluation metrics.
   *
   * @return
   */
  def getShardedEvaluatorTypeColumns(evaluators: Seq[EvaluatorType]): Set[String] =
    evaluators
      .flatMap {
        case shardedEvaluatorType: ShardedEvaluatorType => Some(shardedEvaluatorType.idColumn)
        case _ => None
      }
      .toSet
}

case class ShardedPrecisionAtK(k: Int, override val idColumn: String) extends ShardedEvaluatorType {
  val name = s"PRECISION@$k${ShardedEvaluatorType.shardedEvaluatorIdNameSplitter}$idColumn"
}

object ShardedPrecisionAtK {
  val shardedPrecisionAtKPattern = s"(?i:PRECISION)@(\\d+)${ShardedEvaluatorType.shardedEvaluatorIdNameSplitter}(.*)".r
}

case class ShardedAUC(override val idColumn: String) extends ShardedEvaluatorType {
  val name = s"AUC${ShardedEvaluatorType.shardedEvaluatorIdNameSplitter}$idColumn"
}

object ShardedAUC {
  val shardedAUCPattern = s"(?i:AUC)${ShardedEvaluatorType.shardedEvaluatorIdNameSplitter}(.*)".r
}
