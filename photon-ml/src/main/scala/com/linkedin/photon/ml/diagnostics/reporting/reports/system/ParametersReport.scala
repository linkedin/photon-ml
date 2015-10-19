package com.linkedin.photon.ml.diagnostics.reporting.reports.system

import com.linkedin.photon.ml.Params
import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport

/**
 * Capture command line parameters in a reportable form.
 * @param parameters
 */
case class ParametersReport(parameters: Params) extends LogicalReport
