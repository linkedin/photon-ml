package com.linkedin.photon.ml.optimization

/**
 * @author xazhang
 */
case class MFOptimizationConfiguration(maxNumberIterations: Int, numFactors: Int) {
  override def toString: String = {
    s"maxNumberIterations: $maxNumberIterations\tnumFactors: $numFactors"
  }
}

object MFOptimizationConfiguration {
  def parseAndBuildFromString(string: String): MFOptimizationConfiguration = {
    val Array(maxNumberIterations, numFactors) = string.split(",").map(_.toInt)
    MFOptimizationConfiguration(maxNumberIterations, numFactors)
  }
}
