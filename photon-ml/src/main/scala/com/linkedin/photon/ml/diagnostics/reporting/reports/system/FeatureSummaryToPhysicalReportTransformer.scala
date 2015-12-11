package com.linkedin.photon.ml.diagnostics.reporting.reports.system

import com.linkedin.photon.ml.diagnostics.reporting.reports.Utils
import com.linkedin.photon.ml.diagnostics.reporting.{
  SimpleTextPhysicalReport, BulletedListPhysicalReport, SectionPhysicalReport, LogicalToPhysicalReportTransformer}

/**
 * Convert a feature summary into presentable form.
 */
class FeatureSummaryToPhysicalReportTransformer
  extends LogicalToPhysicalReportTransformer[FeatureSummaryReport, SectionPhysicalReport] {

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

        new SimpleTextPhysicalReport(
          s"(N: [$featureName], T:[$featureTerm]) min: $min, mean: $mean, max: $max, variance: $variance, " +
          s"# non-zero: $nnz / $count")
      }).toSeq
    )

    new SectionPhysicalReport(Seq(contents), SECTION_TITLE)
  }
}

object FeatureSummaryToPhysicalReportTransformer {
  val SECTION_TITLE = "Feature Summary"
}
