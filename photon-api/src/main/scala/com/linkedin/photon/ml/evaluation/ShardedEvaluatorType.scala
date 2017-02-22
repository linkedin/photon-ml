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

/**
 * Trait for sharded evaluator, i.e., evaluator applied on sharded data set.
 */
trait ShardedEvaluatorType extends EvaluatorType {
  /**
   * Type of the id used to shard the data for evaluation, e.g., documentId or queryId
   */
  val idType: String
}

object ShardedEvaluatorType {
  val shardedEvaluatorIdNameSplitter = ":"
}

case class ShardedPrecisionAtK(k: Int, idType: String) extends ShardedEvaluatorType {
  val name = s"PRECISION@$k${ShardedEvaluatorType.shardedEvaluatorIdNameSplitter}$idType"
}

object ShardedPrecisionAtK {
  val shardedPrecisionAtKPattern = s"(?i:PRECISION)@(\\d+)${ShardedEvaluatorType.shardedEvaluatorIdNameSplitter}(.*)".r
}

case class ShardedAUC(idType: String) extends ShardedEvaluatorType {
  val name = s"AUC${ShardedEvaluatorType.shardedEvaluatorIdNameSplitter}$idType"
}

object ShardedAUC {
  val shardedAUCPattern = s"(?i:AUC)${ShardedEvaluatorType.shardedEvaluatorIdNameSplitter}(.*)".r
}
