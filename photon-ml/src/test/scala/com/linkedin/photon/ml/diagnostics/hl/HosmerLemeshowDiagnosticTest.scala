/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
 * Exercise majority of [[HosmerLemeshowDiagnostic]], focus here is primarily on the sanity of the results computation.
 * The spark interactions / plumbing are sequestered in the integration tests
 *
 */
class HosmerLemeshowDiagnosticTest {

  import HosmerLemeshowDiagnosticTest._
  import org.testng.Assert._

  @DataProvider
  def generateHappyPathTestCases(): Array[Array[Any]] = {
    val checkOther: HosmerLemeshowReport => Unit = (x: HosmerLemeshowReport) => {
      println(s"Checking report:\n$x")
      assertEquals(x.binningMsg, MockPredictedProbabilityVersusObservedFrequencyBinner.BINNING_MESSAGE, "Binning message matches expectations")
      assertFalse(x.chiSquaredScore.isInfinite, "Computed score is not infinite")
      assertFalse(x.chiSquaredScore.isNaN, "Computed score is not a NaN")
      assertFalse(x.chiSquaredProb.isInfinite, "Chi^2 probability is not infinite")
      assertFalse(x.chiSquaredProb.isNaN, "Chi^2 probability is not a NAN")
      assertTrue(x.chiSquaredProb >= 0.0, s"Computed Chi^2 prob [${x.chiSquaredProb}] is non-negative")
      assertTrue(x.chiSquaredProb <= 1.0, s"Computed Chi^2 prob [${x.chiSquaredProb}] is no more than 1")
      assertTrue(x.chiSquaredScore >= 0, "Chi square score is positive")
      assertEquals(x.standardConfidencesAndCutoffs.length, HosmerLemeshowDiagnostic.STANDARD_CONFIDENCE_LEVELS.length, "Number of (confidence level, chi^2 cutoff) matches expected")
      var prevProb: Double = -1.0
      var prevCut: Double = -1.0
      x.standardConfidencesAndCutoffs.foreach(x => {
        assertTrue(x._1 > prevProb, s"Confidence [${x._1}] is greater than previous [$prevProb]")
        assertTrue(x._2 > prevCut, s"Cutoff [${x._2}] is greater than previous [$prevCut]")
        prevProb = x._1
        prevCut = x._2
      })
      assertEquals(x.degreesOfFreedom, x.histogram.length - 2, "Degrees of freedom match expectations")
    }

    val checkStrictlyPositive: HosmerLemeshowReport => Unit = (x: HosmerLemeshowReport) => {
      checkOther(x)
      assertTrue(x.chiSquaredScore > 0.0, s"Chi^2 [${x.chiSquaredScore}] > 0")
    }

    val checkEmptyBin: HosmerLemeshowReport => Unit = (x: HosmerLemeshowReport) => {
      checkStrictlyPositive(x)
      assertTrue(x.chiSquareCalculationMsg.toLowerCase.contains("is too small to soundly use"))
    }

    val checkPerfect: HosmerLemeshowReport => Unit = (x: HosmerLemeshowReport) => {
      checkOther(x)
      assertEquals(x.chiSquaredScore, 0.0, CHI_SQUARED_SCORE_TOLERANCE, "Chi^2 on perfect data is zero")
    }

    val allGenerators: Array[(String, Int => Array[PredictedProbabilityVersusObservedFrequencyHistogramBin], HosmerLemeshowReport => Unit)] = Array(
      ("Perfect", (x: Int) => generatePerfect(x), checkPerfect),
      ("Inverted", (x: Int) => generateInvertedPerfect(x), checkStrictlyPositive),
      ("All positive", (x: Int) => generateAllPos(x), checkStrictlyPositive),
      ("All negative", (x: Int) => generateAllNeg(x), checkStrictlyPositive),
      ("Empty bin", (x: Int) => generateEmptyBin(x), checkEmptyBin))

    val allCases: List[Array[Any]] = for {bins <- NUMBER_OF_SAMPLES; scenario <- allGenerators} yield Array[Any](s"${scenario._1} with $bins bins", new MockPredictedProbabilityVersusObservedFrequencyBinner(scenario._2(bins)), scenario._3)
    allCases.toArray
  }

  @Test(dataProvider = "generateHappyPathTestCases")
  def checkHLHappyPath(desc: String, binner: PredictedProbabilityVersusObservedFrequencyBinner, reportValidator: HosmerLemeshowReport => Unit): Unit = {
    runHosmerLemeshow(binner, reportValidator)
  }

  def runHosmerLemeshow(binner: PredictedProbabilityVersusObservedFrequencyBinner, reportValidator: HosmerLemeshowReport => Unit): Unit = {
    val diagnostic = new HosmerLemeshowDiagnostic(binner)
    val report = diagnostic.diagnose(0, 0L, null)
    reportValidator(report)
  }
}

object HosmerLemeshowDiagnosticTest {
  val SAMPLES_PER_BIN: Long = 10000
  val NUMBER_OF_SAMPLES: List[Int] = List(3, 100, 101)
  val CHI_SQUARED_SCORE_TOLERANCE: Double = 1e-6

  def generatePerfect(numBins: Int): Array[PredictedProbabilityVersusObservedFrequencyHistogramBin] = {
    val bins = AbstractPredictedProbabilityVersusObservedFrequencyBinner.generateInitialBins(numBins)

    bins.map(x => {
      val avgProb: Double = (x.lowerBound + x.upperBound) / 2.0
      val numPos: Long = math.ceil(avgProb * SAMPLES_PER_BIN).toLong
      val numNeg: Long = SAMPLES_PER_BIN - numPos
      new PredictedProbabilityVersusObservedFrequencyHistogramBin(x.lowerBound, x.upperBound, numPos, numNeg)
    })
  }

  def generateInvertedPerfect(numBins: Int): Array[PredictedProbabilityVersusObservedFrequencyHistogramBin] = {
    val bins = AbstractPredictedProbabilityVersusObservedFrequencyBinner.generateInitialBins(numBins)

    bins.map(x => {
      val avgProb: Double = (x.lowerBound + x.upperBound) / 2.0
      val numPos: Long = math.ceil(avgProb * SAMPLES_PER_BIN).toLong
      val numNeg: Long = SAMPLES_PER_BIN - numPos
      new PredictedProbabilityVersusObservedFrequencyHistogramBin(x.lowerBound, x.upperBound, numNeg, numPos)
    })
  }

  def generateAllPos(numBins: Int): Array[PredictedProbabilityVersusObservedFrequencyHistogramBin] = {
    val bins = AbstractPredictedProbabilityVersusObservedFrequencyBinner.generateInitialBins(numBins)

    bins.map(x => {
      val numPos: Long = SAMPLES_PER_BIN
      val numNeg: Long = 0
      new PredictedProbabilityVersusObservedFrequencyHistogramBin(x.lowerBound, x.upperBound, numPos, numNeg)
    })
  }

  def generateAllNeg(numBins: Int): Array[PredictedProbabilityVersusObservedFrequencyHistogramBin] = {
    val bins = AbstractPredictedProbabilityVersusObservedFrequencyBinner.generateInitialBins(numBins)

    bins.map(x => {
      val numPos: Long = 0
      val numNeg: Long = SAMPLES_PER_BIN
      new PredictedProbabilityVersusObservedFrequencyHistogramBin(x.lowerBound, x.upperBound, numPos, numNeg)
    })
  }

  def generateEmptyBin(numBins: Int): Array[PredictedProbabilityVersusObservedFrequencyHistogramBin] = {
    val bins = AbstractPredictedProbabilityVersusObservedFrequencyBinner.generateInitialBins(numBins)

    bins.map(x => {
      val numPos: Long = 0
      val numNeg: Long = if (x.lowerBound == 0.0) 0L else SAMPLES_PER_BIN
      new PredictedProbabilityVersusObservedFrequencyHistogramBin(x.lowerBound, x.upperBound, numPos, numNeg)
    })
  }
}