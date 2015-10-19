package com.linkedin.photon.ml.stat

import breeze.linalg.Vector
import org.apache.spark.mllib.linalg.VectorsWrapper
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary


/**
 * A wrapper of [[https://spark.apache.org/docs/1.4.0/api/scala/index.html#org.apache.spark.mllib.stat.MultivariateStatisticalSummary MultivariateStatisticalSummary]]
 * of mllib to use breeze vectors instead of mllib vectors.
 * The summary provides mean, variance, max, min, normL1 and normL2 for each features.
 *
 * @author dpeng
 */
case class BasicStatisticalSummary(mean: Vector[Double],
                                   variance: Vector[Double],
                                   count: Long,
                                   numNonzeros: Vector[Double],
                                   max: Vector[Double],
                                   min: Vector[Double],
                                   normL1: Vector[Double],
                                   normL2: Vector[Double])

object BasicStatisticalSummary {
  /**
   * Convert a [[https://spark.apache.org/docs/1.4.0/api/scala/index.html#org.apache.spark.mllib.stat.MultivariateStatisticalSummary MultivariateStatisticalSummary]]
   * of mllib to a case instance with breeze vectors.
   *
   * @param mllibSummary Summary from mllib
   * @return The summary with breeze vectors
   */
  def apply(mllibSummary: MultivariateStatisticalSummary): BasicStatisticalSummary = {
    val tMean = VectorsWrapper.mllibToBreeze(mllibSummary.mean)
    val tVariance = VectorsWrapper.mllibToBreeze(mllibSummary.variance)
    val tNumNonZeros = VectorsWrapper.mllibToBreeze(mllibSummary.numNonzeros)
    val tMax = VectorsWrapper.mllibToBreeze(mllibSummary.max)
    val tMin = VectorsWrapper.mllibToBreeze(mllibSummary.min)
    val tNormL1 = VectorsWrapper.mllibToBreeze(mllibSummary.normL1)
    val tNormL2 = VectorsWrapper.mllibToBreeze(mllibSummary.normL2)
    this(tMean, tVariance, mllibSummary.count, tNumNonZeros, tMax, tMin, tNormL1, tNormL2)
  }
}
