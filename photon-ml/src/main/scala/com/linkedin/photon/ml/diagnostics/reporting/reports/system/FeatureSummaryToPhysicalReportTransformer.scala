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
package com.linkedin.photon.ml.diagnostics.reporting.reports.system

import com.linkedin.photon.ml.diagnostics.reporting.reports.Utils
import com.linkedin.photon.ml.diagnostics.reporting.{SimpleTextPhysicalReport, BulletedListPhysicalReport, SectionPhysicalReport, LogicalToPhysicalReportTransformer}

/**
 * Convert a feature summary into presentable form.
 */
class FeatureSummaryToPhysicalReportTransformer extends LogicalToPhysicalReportTransformer[FeatureSummaryReport, SectionPhysicalReport] {

  import FeatureSummaryToPhysicalReportTransformer._

  def transform(sum: FeatureSummaryReport): SectionPhysicalReport = {
    val contents = new BulletedListPhysicalReport(
      sum.nameToIndex.map(x => {
        val (featureName, featureTerm) = Utils.extractNameTerm(x._1)
        val min = sum.summary.min(x._2)
        val max = sum.summary.max(x._2)
        val mean = sum.summary.mean(x._2)
        val variance = sum.summary.variance(x._2)
        val nnz = sum.summary.numNonzeros(x._2)
        val count = sum.summary.count

        new SimpleTextPhysicalReport(s"(N: [$featureName], T:[$featureTerm]) min: $min, mean: $mean, max: $max, variance: $variance, # non-zero: $nnz / $count")
      }).toSeq
    )

    new SectionPhysicalReport(Seq(contents), SECTION_TITLE)
  }
}

object FeatureSummaryToPhysicalReportTransformer {
  val SECTION_TITLE = "Feature Summary"
}
