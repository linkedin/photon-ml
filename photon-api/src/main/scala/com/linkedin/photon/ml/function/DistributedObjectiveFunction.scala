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
package com.linkedin.photon.ml.function

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint

/**
 * The base objective function used by DistributedOptimizationProblems. This function works with an RDD of data
 * distributed across the cluster.
 *
 * @param treeAggregateDepth The depth used by treeAggregate. Depth 1 indicates normal linear aggregate. Using
 *                           depth > 1 can reduce memory consumption in the Driver and may also speed up the
 *                           aggregation.
 */
abstract class DistributedObjectiveFunction(treeAggregateDepth: Int) extends ObjectiveFunction {

  type Data = RDD[LabeledPoint]

  require(treeAggregateDepth > 0, s"Tree aggregate depth must be greater than 0: $treeAggregateDepth")
}
