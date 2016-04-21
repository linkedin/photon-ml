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
package com.linkedin.photon.ml.optimization.game

import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.optimization.RegularizationType.RegularizationType

/**
 * Configuration object for GLM optimization
 *
 * @param maxNumberIterations maximum number of iterations
 * @param convergenceTolerance convergence tolerance
 * @param regularizationWeight regularization weight
 * @param downSamplingRate downsampling rate
 * @param optimizerType optimizer type (e.g. LBFGS, TRON)
 * @param regularizationType regularization type
 * @author xazhang
 */
protected[ml] case class GLMOptimizationConfiguration private (
    maxNumberIterations: Int = 20,
    convergenceTolerance: Double = 1e-5,
    regularizationWeight: Double = 50,
    downSamplingRate: Double = 1,
    optimizerType: OptimizerType = OptimizerType.TRON,
    regularizationType: RegularizationType = RegularizationType.L2) {

  override def toString: String = {
    s"maxNumberIterations: $maxNumberIterations\t" +
      s"convergenceTolerance: $convergenceTolerance\t" +
      s"regularizationWeight: $regularizationWeight\t" +
      s"downSamplingRate: $downSamplingRate\t" +
      s"optimizerType: $optimizerType\t" +
      s"regularizationType: $regularizationType"
  }
}

object GLMOptimizationConfiguration {
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
   * @param string the string representation
   * @todo Add assert and meaningful parsing error message here
   */
  protected[ml] def parseAndBuildFromString(string: String): GLMOptimizationConfiguration = {
    val Array(maxNumberIterationsStr, convergenceToleranceStr, regularizationWeightStr, downSamplingRateStr,
    optimizerTypeStr, regularizationTypeStr) = string.split(",")
    val maxNumberIterations = maxNumberIterationsStr.toInt
    val convergenceTolerance = convergenceToleranceStr.toDouble
    val regularizationWeight = regularizationWeightStr.toDouble
    val downSamplingRate = downSamplingRateStr.toDouble

    assert(downSamplingRate > 0.0 && downSamplingRate <= 1.0, s"Unexpected downSamplingRate: $downSamplingRate")
    val optimizerType = OptimizerType.withName(optimizerTypeStr.toUpperCase)
    val regularizationType = RegularizationType.withName(regularizationTypeStr.toUpperCase)
    GLMOptimizationConfiguration(maxNumberIterations, convergenceTolerance, regularizationWeight, downSamplingRate,
      optimizerType, regularizationType)
  }
}
