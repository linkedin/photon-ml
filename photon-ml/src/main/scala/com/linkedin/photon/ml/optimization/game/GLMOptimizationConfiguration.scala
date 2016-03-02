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
case class GLMOptimizationConfiguration(
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
   * Parse and build the configuration object from a string representation
   *
   * @param string the string representation
   * @todo Add assert and meaningful parsing error message here
   */
  def parseAndBuildFromString(string: String): GLMOptimizationConfiguration = {
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
