package com.linkedin.photon.ml.stat

import breeze.linalg.Vector
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.VectorsWrapper
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary


/**
 * A wrapper of [[https://spark.apache.org/docs/1.4.0/api/scala/index.html#org.apache.spark.mllib.stat.MultivariateStatisticalSummary MultivariateStatisticalSummary]]
 * of mllib to use breeze vectors instead of mllib vectors.
 * The summary provides mean, variance, max, min, normL1 and normL2 for each features, as well as the expected magnitude of features (meanAbs) to assist in computing
 * feature importance.
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
                                   normL2: Vector[Double],
                                   meanAbs: Vector[Double])

object BasicStatisticalSummary extends Logging {
  /**
   * Convert a [[https://spark.apache.org/docs/1.4.0/api/scala/index.html#org.apache.spark.mllib.stat.MultivariateStatisticalSummary MultivariateStatisticalSummary]]
   * of mllib to a case instance with breeze vectors.
   *
   * @param mllibSummary Summary from mllib
   * @return The summary with breeze vectors
   */
  def apply(mllibSummary: MultivariateStatisticalSummary, meanAbs: Vector[Double]): BasicStatisticalSummary = {
    val tMean = VectorsWrapper.mllibToBreeze(mllibSummary.mean)
    val tVariance = VectorsWrapper.mllibToBreeze(mllibSummary.variance)
    val tNumNonZeros = VectorsWrapper.mllibToBreeze(mllibSummary.numNonzeros)
    val tMax = VectorsWrapper.mllibToBreeze(mllibSummary.max)
    val tMin = VectorsWrapper.mllibToBreeze(mllibSummary.min)
    val tNormL1 = VectorsWrapper.mllibToBreeze(mllibSummary.normL1)
    val tNormL2 = VectorsWrapper.mllibToBreeze(mllibSummary.normL2)

    val adjustedCount = tVariance.activeIterator.foldLeft(0)((count, idxValuePair) => {
      if (idxValuePair._2.isNaN || idxValuePair._2.isInfinite || idxValuePair._2 < 0) {
        logWarning(s"Detected invalid variance at index ${idxValuePair._1} (${idxValuePair._2})")
        count + 1
      } else {
        count
      }
    })

    if (adjustedCount > 0) {
      logWarning(s"Found $adjustedCount features where variance was either non-positive, not-a-number, or infinite. The variances for these features have been re-set to 1.0.")
    }

    this(tMean, tVariance.mapActiveValues(x => if (x.isNaN || x.isInfinite || x < 0) 1.0 else x), mllibSummary.count, tNumNonZeros, tMax, tMin, tNormL1, tNormL2, meanAbs)
  }
}
