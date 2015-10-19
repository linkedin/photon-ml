package com.linkedin.photon.ml.diagnostics.hl

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport

/**
 * Capture the results of a H-L test.
 *
 * @param binningMsg
 * Any messages generated during the binning process
 *
 * @param chiSquareCalculationMsg
 * Any messages generated during the &Chi;<sup>2</sup> statistic calculation process
 * @param degreesOfFreedom
 * Number of degrees of freedom for the assumed &Chi;<sup>2</sup> distribution
 *
 * @param chiSquaredScore
 * Computed &Chi;<sup>2</sup> statistic
 *
 * @param chiSquaredProb
 * Pr[x &lt; &Chi;<sup>2</sup>] (i.e. the probability that this score is by chance
 * alone)
 *
 * @param standardConfidencesAndCutoffs
 * List of (confidence level, &Chi;<sup>2</sup> cutoff) to help contextualize
 * output
 * @param histogram
 * Observed positive rate binned by expected positive rate
 */
class HosmerLemeshowReport(val binningMsg: String,
                           val chiSquareCalculationMsg: String,
                           val chiSquaredScore: Double,
                           val degreesOfFreedom: Int,
                           val chiSquaredProb: Double,
                           val standardConfidencesAndCutoffs: List[(Double, Double)],
                           val histogram: Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]) extends LogicalReport {

  override def toString(): String = {
    s"${getTestDescription}\n${getPointProbabilityAnalysis}\nCutoffs:\n${getCutoffAnalysis.mkString("\n")}\nHistogram:\n    ${histogram.mkString("\n    ")}"
  }

  def getTestDescription(): String = {
    f"Chi^2 = [$chiSquaredScore%.06f] on [$degreesOfFreedom] degrees of freedom"
  }

  def getPointProbabilityAnalysis(): String = {
    f"Pr[Chi^2 < $chiSquaredScore] = [$chiSquaredProb%.09g]"
  }

  def getCutoffAnalysis(): List[String] = {
    standardConfidencesAndCutoffs.map(x => {
      val reject = if (chiSquaredScore > x._2) "reject" else "accept"
      f"  Pr[X <= ${x._2}%12.09f] = ${100.0 * x._1}%.03f H0 (Well-specified model with Chi^2 <= $chiSquaredScore by chance alone): $reject"
    })
  }
}
