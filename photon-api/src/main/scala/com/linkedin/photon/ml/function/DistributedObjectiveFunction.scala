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

import breeze.linalg.Vector
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import com.linkedin.photon.ml.data.LabeledPoint

/**
 * The base objective function used by DistributedOptimizationProblems. This function works with an RDD of data
 * distributed across the cluster.
 *
 * @param treeAggregateDepth The depth used by treeAggregate. Depth 1 indicates normal linear aggregate. Using
 *                           depth > 1 can reduce memory consumption in the Driver and may also speed up the
 *                           aggregation. It is experimental currently because treeAggregate is unstable in Spark
 *                           versions 1.4 and 1.5.
 */
abstract class DistributedObjectiveFunction(treeAggregateDepth: Int) extends ObjectiveFunction {

  type Data = RDD[LabeledPoint]
  type Coefficients = Broadcast[Vector[Double]]

  require(treeAggregateDepth > 0, s"Tree aggregate depth must be greater than 0: $treeAggregateDepth")

  private lazy val sc: SparkContext = SparkSession.builder().getOrCreate().sparkContext

  /**
   * Compute the size of the domain for the given input data (i.e. the number of features, including the intercept if
   * there is one).
   *
   * @param input The given data for which to compute the domain dimension
   * @return The domain dimension
   */
  override protected[ml] def domainDimension(input: Data): Int = input.first.features.size

  /**
   * DistributedOptimizationProblems compute objective value over an RDD distributed across several tasks over one or
   * more executors. Thus, DistributedObjectiveFunction expects broadcasted coefficients to reduce network overhead.
   *
   * @param coefficients A coefficients Vector to convert
   * @return A broadcast of the given coefficients Vector
   */
  override protected[ml] def convertFromVector(coefficients: Vector[Double]): Coefficients =
    sc.broadcast(coefficients)

  /**
   * DistributedObjectiveFunctions handle broadcasted Vectors. Fetch the underlying Vector.
   *
   * @param coefficients A broadcasted coefficients vector
   * @return The underlying coefficients Vector
   */
  override protected[ml] def convertToVector(coefficients: Coefficients): Vector[Double] = coefficients.value

  /**
   * Unpersist the coefficients broadcast by convertFromVector.
   *
   * @param coefficients The broadcast coefficients to unpersist
   */
  override protected[ml] def cleanupCoefficients(coefficients: Coefficients): Unit = coefficients.unpersist()
}
