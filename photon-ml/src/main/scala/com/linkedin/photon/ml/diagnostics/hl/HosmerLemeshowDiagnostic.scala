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

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.diagnostics.ModelDiagnostic
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import org.apache.commons.math3.distribution.{ChiSquaredDistribution, RealDistribution}
import org.apache.spark.rdd.RDD

/**
 * Implements the Hosmer-Lemeshow goodness-of-fit test for LR problems
 * (see here: http://thestatsgeek.com/2014/02/16/the-hosmer-lemeshow-goodness-of-fit-test-for-logistic-regression/)
 */
class HosmerLemeshowDiagnostic(scoreBinner: PredictedProbabilityVersusObservedFrequencyBinner = new DefaultPredictedProbabilityVersusObservedFrequencyBinner()) extends ModelDiagnostic[LogisticRegressionModel, HosmerLemeshowReport] {

  import HosmerLemeshowDiagnostic._

  def diagnose(model: LogisticRegressionModel, data: RDD[LabeledPoint], summary: Option[BasicStatisticalSummary]): HosmerLemeshowReport = {
    val scored = data.map(x => (x.label, model.computeMeanFunction(x.features)))
    val count = data.count
    val dim = data.first.features.size
    diagnose(dim, count, scored)
  }

  def diagnose(numDimensions: Int, numSamples: Long, observedVScored: RDD[(Double, Double)]): HosmerLemeshowReport = {
    val (binMsg: String, binnedScores: Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]) = scoreBinner.bin(numSamples, numDimensions, observedVScored)

    val (chiSquaredMsg, chiSquaredScore) = binnedScores.map(bin => {
      val msg: StringBuilder = new StringBuilder()

      val deltaPos: Double = if (bin.expectedPosCount > 0) {
        (bin.observedPosCount - bin.expectedPosCount) * (bin.observedPosCount - bin.expectedPosCount) / bin.expectedPosCount.toDouble
      } else {
        0.0
      }

      if (bin.expectedPosCount < MINIMUM_EXPECTED_IN_BUCKET) {
        msg.append(s"For bin [$bin], expected positive count is too small to soundly use in a Chi^2 estimate\n")
      }

      val deltaNeg: Double = if (bin.expectedNegCount > 0) {
        (bin.observedNegCount - bin.expectedNegCount) * (bin.observedNegCount - bin.expectedNegCount) / bin.expectedNegCount.toDouble
      } else {
        0.0
      }

      if (bin.expectedNegCount < MINIMUM_EXPECTED_IN_BUCKET) {
        msg.append(s"For bin [$bin], expected negative count is too small to soundly use in a Chi^2 estimate\n")
      }

      (msg.toString, deltaPos + deltaNeg)
    }).foldLeft(("", 0.0))((a, b) => (a._1 + b._1, a._2 + b._2))

    // Step 3: now do the Chi^2 test and update the report with relevant numbers (e.g. computed Chi^2, d.o.f, probability /
    //         loglik of such an extreme result, cutoffs for 90, 95, 99, 99.999999% likelihood)
    val degressOfFreedom: Int = binnedScores.size - 2
    val chiSquareDF: RealDistribution = new ChiSquaredDistribution(degressOfFreedom)
    val cutoffs = HosmerLemeshowDiagnostic.STANDARD_CONFIDENCE_LEVELS.map(x => (x, chiSquareDF.inverseCumulativeProbability(x)))
    val probAtChiSquare = chiSquareDF.cumulativeProbability(chiSquaredScore)
    new HosmerLemeshowReport(binMsg, chiSquaredMsg, chiSquaredScore, degressOfFreedom, probAtChiSquare, cutoffs, binnedScores)
  }
}

object HosmerLemeshowDiagnostic {
  val STANDARD_CONFIDENCE_LEVELS: List[Double] = List(0.000001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999999)
  val MINIMUM_EXPECTED_IN_BUCKET: Int = 5
}
