package com.linkedin.photon.ml.optimization.game

/**
 * Configuration object for matrix factorization optimization
 *
 * @param maxNumberIterations maximum number of iterations
 * @param numFactors number of factors
 * @author xazhang
 */
case class MFOptimizationConfiguration(maxNumberIterations: Int, numFactors: Int) {
  override def toString: String = {
    s"maxNumberIterations: $maxNumberIterations\tnumFactors: $numFactors"
  }
}

object MFOptimizationConfiguration {

  /**
   * Parse and build the configuration object from a string representation
   *
   * @param string the string representation
   * @todo Add assert and meaningful parsing error message here
   */
  def parseAndBuildFromString(string: String): MFOptimizationConfiguration = {
    val Array(maxNumberIterations, numFactors) = string.split(",").map(_.toInt)
    MFOptimizationConfiguration(maxNumberIterations, numFactors)
  }
}
