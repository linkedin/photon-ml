package com.linkedin.photon.ml.diagnostics.reporting.reports.combined

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport
import com.linkedin.photon.ml.diagnostics.reporting.reports.model.ModelDiagnosticReport
import com.linkedin.photon.ml.diagnostics.reporting.reports.system.SystemReport
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * A complete model diagnostic report.
 *
 * @param systemReport
 *                     System-level information to include in the report
 * @param modelReports
 *                     Model-level information to include in the report
 */
case class DiagnosticReport(
    val systemReport: SystemReport,
    val modelReports: scala.collection.mutable.ListBuffer[ModelDiagnosticReport[GeneralizedLinearModel]])
  extends LogicalReport
