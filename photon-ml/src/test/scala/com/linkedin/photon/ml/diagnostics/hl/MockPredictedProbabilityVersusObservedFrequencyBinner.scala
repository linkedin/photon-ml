package com.linkedin.photon.ml.diagnostics.hl

import org.apache.spark.rdd.RDD

/**
 * Mock for injecting specific O/E histograms into Hosmer-Lemeshow diagnostic
 */
class MockPredictedProbabilityVersusObservedFrequencyBinner(val bins: Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]) extends PredictedProbabilityVersusObservedFrequencyBinner {
  def bin(numItems: Long, numDimensions: Int, observedVExpected: RDD[(Double, Double)]): (String, Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]) = {
    (MockPredictedProbabilityVersusObservedFrequencyBinner.BINNING_MESSAGE, bins)
  }
}

object MockPredictedProbabilityVersusObservedFrequencyBinner {
  val BINNING_MESSAGE = "Mock"
}
