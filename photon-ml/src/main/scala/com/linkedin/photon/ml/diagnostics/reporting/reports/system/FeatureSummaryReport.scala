package com.linkedin.photon.ml.diagnostics.reporting.reports.system

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport
import com.linkedin.photon.ml.stat.BasicStatisticalSummary

/**
 * Report describing a feature summary
 * @param nameToIndex
 *                    Eventually a map of (featureId &rarr; index); for now, a map of a string encoding (key/term) to
 *                    feature index.
 * @param summary
 *                Feature summary
 */
case class FeatureSummaryReport(
  nameToIndex: Map[String, Int],
  summary:BasicStatisticalSummary) extends LogicalReport
