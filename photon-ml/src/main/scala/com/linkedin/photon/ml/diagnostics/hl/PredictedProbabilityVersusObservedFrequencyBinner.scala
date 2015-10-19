package com.linkedin.photon.ml.diagnostics.hl

import org.apache.spark.rdd.RDD

/**
 * Generic interface for binning logistic-regression type responses into a list of (prediction range -> actual counts v expected counts) tuples
 * It is possible, in the future, that it would make sense to further refine / abstract this class if we think about doing general &Chi;<sup>2</sup>
 * testing or similar goodness-of-fit measures.
 *
 * For now, the major purpose of this class is to allow me to mock this logic out for testing.
 */
trait PredictedProbabilityVersusObservedFrequencyBinner {
  def bin(numItems: Long, numDimensions: Int, observedVExpected: RDD[(Double, Double)]): (String, Array[PredictedProbabilityVersusObservedFrequencyHistogramBin])
}
