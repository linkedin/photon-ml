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
package com.linkedin.photon.ml.optimization.game

import scala.util.parsing.json.JSON

import com.linkedin.photon.ml.optimization._

/**
 * Configuration object for GLM optimization
 *
 * @param optimizerConfig Optimizer configuration
 * @param regularizationContext Regularization context
 * @param regularizationWeight Regularization weight
 * @param downSamplingRate Down-sampling rate
 */
case class GLMOptimizationConfiguration (
    optimizerConfig: OptimizerConfig = OptimizerConfig(OptimizerType.TRON, 20, 1E-5, None),
    regularizationContext: RegularizationContext = NoRegularizationContext,
    regularizationWeight: Double = 0D,
    downSamplingRate: Double = 1D) {

  checkInvariants()

  /**
   * The invariants that hold for any GLMOptimizationConfiguration.
   *
   * OptimizerConfig and RegularizationContext have their own checkInvariants called from their respective
   * constructors.
   */
  def checkInvariants(): Unit = {
    require(0 <= regularizationWeight, s"Negative regularization weight: $regularizationWeight")
    require(downSamplingRate > 0.0 && downSamplingRate <= 1.0, s"Unexpected downSamplingRate: $downSamplingRate")
  }

  override def toString: String =
    s"optimizerConfig: ${optimizerConfig.toSummaryString}, " +
      s"regularizationContext: ${regularizationContext.toSummaryString}, " +
      s"regularizationWeight: $regularizationWeight, " +
      s"downSamplingRate: $downSamplingRate"

  def toJson: String =
    s"""{
       |   "optimizerConfig": ${optimizerConfig.toJson},
       |   "regularizationContext": ${regularizationContext.toJson},
       |   "regularizationWeight": $regularizationWeight,
       |   "downSamplingRate": $downSamplingRate
       |}""".stripMargin
}

object GLMOptimizationConfiguration {

  protected[ml] val SPLITTER = ","
  protected[ml] val EXPECTED_FORMAT: String =
    s"maxNumberIterations${SPLITTER}convergenceTolerance${SPLITTER}regularizationWeight$SPLITTER" +
      s"downSamplingRate${SPLITTER}optimizerType${SPLITTER}regularizationType"
  protected[ml] val EXPECTED_NUM_CONFIGS = 6

  /**
   * Parse and build the configuration object from a string representation.
   * The string is expected to be a comma separated list with order of elements being
   * <ol>
   *  <li> Maximum number of iterations
   *  <li> Convergence tolerance
   *  <li> Regularization weight
   *  <li> Down-sampling rate
   *  <li> Optimizer Type
   *  <li> Regularization type
   * </ol>
   *
   * @note this is still useful when coming from the templates, although we can read from JSON
   *
   * @param string The string representation
   */
  protected[ml] def parseAndBuildFromString(string: String): GLMOptimizationConfiguration = {

    val configParams: Array[String] = string.split(SPLITTER).map(_.trim)

    require(configParams.length == EXPECTED_NUM_CONFIGS,
      s"Parsing $string failed! The expected GLM optimization configuration should contain $EXPECTED_NUM_CONFIGS " +
        s"parts separated by \'$SPLITTER\', but found ${configParams.length}. Expected format: $EXPECTED_FORMAT")

    val Array(maxNumberIterationsStr,
      convergenceToleranceStr,
      regularizationWeightStr,
      downSamplingRateStr,
      optimizerTypeStr,
      regularizationTypeStr) = configParams

    val maxNumberIterations = maxNumberIterationsStr.toInt
    val convergenceTolerance = convergenceToleranceStr.toDouble
    val regularizationWeight = regularizationWeightStr.toDouble
    val downSamplingRate = downSamplingRateStr.toDouble
    val optimizerType = OptimizerType.withName(optimizerTypeStr.toUpperCase)
    val regularizationContext = RegularizationType.withName(regularizationTypeStr.toUpperCase) match {
      case RegularizationType.NONE => NoRegularizationContext
      case RegularizationType.L1 => L1RegularizationContext
      case RegularizationType.L2 => L2RegularizationContext
      // TODO: Elastic Net regularization
      case other => throw new UnsupportedOperationException(s"Regularization of type $other is not supported.")
    }

    val optimizerConfig = OptimizerConfig(optimizerType, maxNumberIterations, convergenceTolerance, None)

    GLMOptimizationConfiguration(optimizerConfig, regularizationContext, regularizationWeight, downSamplingRate)
  }

  /**
   * Same as parseAndBuildFromString, but makes code more readable.
   *
   * @param string A String representation of a GLMOptimizationConfiguration
   * @return An instance of GLMOptimizationConfiguration
   */
  def apply(string: String): GLMOptimizationConfiguration = parseAndBuildFromString(string)

  /**
   * A factory method from JSON format.
   *
   * @param jsonString The JSON string to parse and create an instance from
   * @return An instance of GLMOptimizationConfiguration if the JSON string was parsed successully, None otherwise
   */
  def fromJson(jsonString: String): Option[GLMOptimizationConfiguration] = {

    // Upon JSON.parseFull return, obj looks like:
    // Some(Map(optimizerConfig -> Map(optimizerType -> TRON, maximumIterations -> 10.0, tolerance -> 0.01),
    //          regularizationContext -> Map(regularizationType -> L2, elasticNetParams -> List()),
    //          regularizationWeight -> 1.0,
    //          downSamplingRate -> 0.3))
    val obj = JSON.parseFull(jsonString)

    obj map {
      case mm: Any =>
        val m = mm.asInstanceOf[Map[String, Any]]
        val optimizerConfig = OptimizerConfig(m("optimizerConfig").asInstanceOf[Map[String, Any]])
        val regularizationContext = RegularizationContext(m("regularizationContext").asInstanceOf[Map[String, Any]])
        val regularizationWeight = m("regularizationWeight").asInstanceOf[Double]
        val downSamplingRate = m("downSamplingRate").asInstanceOf[Double]
        GLMOptimizationConfiguration(optimizerConfig, regularizationContext, regularizationWeight, downSamplingRate)

      case _ =>
        throw new RuntimeException(s"Can't parse GLMOptimizationConfiguration: $jsonString")
    }
  }
}
