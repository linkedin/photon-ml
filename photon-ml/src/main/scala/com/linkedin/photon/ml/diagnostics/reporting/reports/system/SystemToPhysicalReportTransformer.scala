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