package com.linkedin.photon.ml.diagnostics.hl

import org.testng.annotations.{DataProvider, Test}

/**
 * Created by bdrew on 10/7/15.
 */
class PredictedProbabilityVersusObservedFrequencyHistogramBinTest {

  import org.testng.Assert._

  @DataProvider
  def generateObservedVersusExpectedCases(): Array[Array[Any]] = {
    Array(
      Array(0.0, 1.0, 1000L, 500L, 500L),
      Array(0.0, 0.5, 1000L, 250L, 750L),
      Array(0.5, 1.0, 1000L, 750L, 250L),
      Array(0.0, 0.1, 1000L, 50L, 950L),
      Array(0.9, 1.0, 1000L, 950L, 50L)
    )
  }

  @Test(dataProvider = "generateObservedVersusExpectedCases")
  def checkObservedVersusExpected(minPred: Double, maxPred: Double, numSamples: Long, expectedPos: Long, expectedNeg: Long): Unit = {
    val bin = new PredictedProbabilityVersusObservedFrequencyHistogramBin(minPred, maxPred, numSamples, 0)
    assertEquals(bin.expectedNegCount, expectedNeg, "Computed value of expected negative count matches")
    assertEquals(bin.expectedPosCount, expectedPos, "Computed value of expected positive count matches")
  }
}
