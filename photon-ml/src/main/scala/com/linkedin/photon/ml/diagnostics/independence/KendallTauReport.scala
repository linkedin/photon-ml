package com.linkedin.photon.ml.diagnostics.independence

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport

/**
 * See [[https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient]]
 * @param concordantPairs Count of concordant pairs. Pairs are concordant if ordering by _1 is consistent with _2
 * @param discordantPairs Count of discordant pairs. Pairs are discordant if ordering by _1 is the reverse of ordering
 *   by _2
 * @param numSamples Total number of samples
 * @param totalPairs Total number of pairs (concordant + discordant + other)
 * @param effectivePairs Number of concordant + discordant pairs
 * @param tauAlpha &tau;<sub>&alpha;</sub> statistic
 * @param tauBeta &tau;<sub>&beta;</sub> statistic
 * @param zAlpha z-score for &tau;<sub>&alpha;</sub>
 * @param pValueAlpha p-value for -|zAlpha| (left sided)
 * @param messages Any messages produced while computing these quantities
 */
case class KendallTauReport(val concordantPairs:Long,
                            val discordantPairs:Long,
                            val numSamples:Long,
                            val totalPairs:Long,
                            val effectivePairs:Long,
                            val tauAlpha:Double,
                            val tauBeta:Double,
                            val zAlpha:Double,
                            val pValueAlpha:Double,
                            val messages:String) extends LogicalReport
