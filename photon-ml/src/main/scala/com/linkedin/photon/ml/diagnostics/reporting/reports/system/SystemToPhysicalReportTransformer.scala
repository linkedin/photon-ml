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

import com.linkedin.photon.ml.diagnostics.reporting.{ChapterPhysicalReport, LogicalToPhysicalReportTransformer}

/**
 * Convert a system report into a presentable form.
 */
class SystemToPhysicalReportTransformer extends LogicalToPhysicalReportTransformer[SystemReport, ChapterPhysicalReport] {

  import SystemToPhysicalReportTransformer._

  def transform(sysrep: SystemReport): ChapterPhysicalReport = {
    sysrep match {
      case SystemReport(nameIdx, null, _) =>
        throw new IllegalStateException("System report should, at least, include non-null command line parameters")
      case SystemReport(nameIdx, params, None) =>
        new ChapterPhysicalReport(Seq(PARAMETERS_TRANSFORMER.transform(new ParametersReport(params))), SYSTEM_CHAPTER_HEADER)
      case SystemReport(nameIdx, params, Some(summary)) =>
        new ChapterPhysicalReport(
          Seq(
            PARAMETERS_TRANSFORMER.transform(new ParametersReport(params)),
            SUMMARY_TRANSFORMER.transform(new FeatureSummaryReport(nameIdx, summary))
          ), SYSTEM_CHAPTER_HEADER)

    }
  }
}


object SystemToPhysicalReportTransformer {
  val SYSTEM_CHAPTER_HEADER = "System Settings and Diagnostics"
  val PARAMETERS_TRANSFORMER = new ParametersToPhysicalReportTransformer
  val SUMMARY_TRANSFORMER = new FeatureSummaryToPhysicalReportTransformer
}