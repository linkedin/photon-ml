/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.diagnostics.hl

import org.testng.annotations.{DataProvider, Test}

/**
 * Check AbstractUniformScoreBinner companion object methods
 */
class AbstractPredictedProbabilityVersusObservedFrequencyBinnerTest {

  import org.testng.Assert._

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def checkGenerateInitialBinsBadBins(): Unit = {
    AbstractPredictedProbabilityVersusObservedFrequencyBinner.generateInitialBins(-2)
  }

  @DataProvider(parallel = false)
  def generateNumBins(): Array[Array[Any]] = {
    Array(
      Array(1),
      Array(2),
      Array(3),
      Array(100),
      Array(101)
    )
  }

  @Test(dataProvider = "generateNumBins")
  def checkGenerateInitialBinsGoodBins(expectedNumBins: Int): Unit = {
    val result = AbstractPredictedProbabilityVersusObservedFrequencyBinner.generateInitialBins(expectedNumBins)

    assertEquals(result.length, expectedNumBins, "Got histogram with expected number of bins")

    val numInvalidRange = result.foldLeft(0)((prevCount, histBin) => if (histBin.lowerBound >= histBin.upperBound) prevCount + 1 else prevCount)
    assertEquals(numInvalidRange, 0, "Number of bins where lowerBound >= upperBound is zero")

    var previousUpper: Double = 0.0
    val numInvalidLower = result.foldLeft(0)((prevCount, histBin) => {
      val result = if (histBin.lowerBound != previousUpper) prevCount + 1 else prevCount
      previousUpper = histBin.upperBound
      result
    })

    assertEquals(numInvalidLower, 0, "Number of bins where the present bin's lower bound is not equal to the previous bin's upper bound is zero")
    assertEquals(result(0).lowerBound, 0.0, "Lower bound for the first bin is exactly 0")
    assertEquals(result.last.upperBound, 1.0, "Upper bound for the last bin is exactly 1")
  }

  @Test(dataProvider = "generateNumBins")
  def checkFindBin(expectedNumBins: Int): Unit = {
    val bins = AbstractPredictedProbabilityVersusObservedFrequencyBinner.generateInitialBins(expectedNumBins)

    bins.zipWithIndex.map(binWithIndex => {
      binWithIndex match {
        case (bin: PredictedProbabilityVersusObservedFrequencyHistogramBin, expectedIndex: Int) =>
          val lowerBound = bin.lowerBound
          val lowerBoundBin = AbstractPredictedProbabilityVersusObservedFrequencyBinner.findBin(lowerBound, bins)
          assertEquals(lowerBoundBin, expectedIndex, s"Searching for the lower bound of the bin [$lowerBound] produces that particular bin [$expectedIndex / ${bins.length}]")
          assertTrue(bins(lowerBoundBin).lowerBound <= lowerBound && bins(lowerBoundBin).upperBound > lowerBound, "Lower bound contained in bin")

          val midBound = (bin.lowerBound + bin.upperBound) / 2.0
          val midBoundBin = AbstractPredictedProbabilityVersusObservedFrequencyBinner.findBin(midBound, bins)
          assertEquals(midBoundBin, expectedIndex, "Searching for the mid bound of the bin produces that particular bin")
          assertTrue(bins(midBoundBin).lowerBound <= midBound && bins(midBoundBin).upperBound > midBound, "Mid bound contained in bin")

          val upperBound = bin.upperBound
          val expectedUpperBoundBin = math.min(bins.length - 1, expectedIndex + 1)
          val upperBoundBin = AbstractPredictedProbabilityVersusObservedFrequencyBinner.findBin(upperBound, bins)
          assertEquals(upperBoundBin, expectedUpperBoundBin, "Searching for the upper bound of the bin produces the expected bin")
          assertTrue(bins(upperBoundBin).lowerBound <= upperBound && bins(midBoundBin).upperBound >= upperBound, "Upper bound contained in bin")
      }
    })

  }
}
