package com.linkedin.photon.ml.diagnostics.reporting

/**
 * Denotes a class capable of transforming a particular kind of logical report into a particular kind of
 * physical report.
 *
 * @tparam L
 *           Logical report type
 * @tparam P
 *           Physical report type
 */
trait LogicalToPhysicalReportTransformer[-L <: LogicalReport, +P <:PhysicalReport] {
  def transform(logical:L):P
}
