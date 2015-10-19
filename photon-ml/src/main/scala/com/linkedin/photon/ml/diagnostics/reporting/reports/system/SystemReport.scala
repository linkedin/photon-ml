package com.linkedin.photon.ml.diagnostics.reporting.reports.system

import com.linkedin.photon.ml.Params
import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport
import com.linkedin.photon.ml.stat.BasicStatisticalSummary

/**
 * Composite containing all of the system-related (i.e. common to all models) information.
 *
 * @param nameToIndex
 *                    Mapping of encoded (name, term) tuples &rarr; index
 * @param params
 *               Parameters used to launch the driver
 *
 * @param summary
 *                Computed feature summary
 */
case class SystemReport(val nameToIndex: Map[String, Int], var params: Params, var summary: Option[BasicStatisticalSummary] = None) extends LogicalReport
