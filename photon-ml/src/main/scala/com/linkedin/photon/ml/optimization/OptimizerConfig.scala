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
package com.linkedin.photon.ml.optimization

import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.util.Summarizable

/**
 * Contains configuration information for Optimizer instances.
 */
case class OptimizerConfig(
    optimizerType: OptimizerType,
    maximumIterations: Int,
    tolerance: Double,
    constraintMap: Option[Map[Int, (Double, Double)]])
  extends Summarizable {

  checkInvariants()

  def checkInvariants(): Unit = {
    require(0 <= maximumIterations, s"Less than 1 specified for maximumIterations (specified: $maximumIterations")
    require(0.0d <= tolerance, s"Specified negative tolerance for optimizer: $tolerance")
  }

  // TODO: Add constraintMap to summary and JSON
  override def toSummaryString: String =
    s"optimizerType = $optimizerType, maximumIterations = $maximumIterations, tolerance = $tolerance"

  def toJson: String =
    s"""{
       |   "optimizerType": "$optimizerType",
       |   "maximumIterations": $maximumIterations,
       |   "tolerance": $tolerance
       |}""".stripMargin
}

object OptimizerConfig {

  /**
   * A factory method from a Map, usually in the context of parsing JSON in GLMOptimizationConfiguration.
   *
   * @param m A Map that contains (key, values) for an OptimizerConfig instance's fields
   * @return An instance of OptimizerConfig
   */
  def apply(m: Map[String, Any]): OptimizerConfig =
    new OptimizerConfig(
      OptimizerType.withName(m("optimizerType").asInstanceOf[String]),
      m("maximumIterations").asInstanceOf[Double].toInt, // scala JSON does not parse Int
      m("tolerance").asInstanceOf[Double],
      None)
}
