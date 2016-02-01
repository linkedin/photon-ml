/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.diagnostics.independence

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport

/**
 * See [[https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient]]
 * @param concordantPairs Count of concordant pairs. Pairs are concordant if ordering by _1 is consistent with _2
 * @param discordantPairs Count of discordant pairs. Pairs are discordant if ordering by _1 is the reverse of ordering by _2
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
