package com.linkedin.photon.ml.diagnostics.reporting.text

import com.linkedin.mlease.spark.diagnostics.reporting.PhysicalReport
import com.linkedin.photon.ml.diagnostics.reporting.{PhysicalReport, SpecificRenderer}

/**
 * Simple rendering using .toString
 */
class AnyToStringRenderer[-P <: PhysicalReport] extends SpecificRenderer[P, String] {
  def render(p: P): String = s"ID: [${p.getId}] TYPE: [${p.getClass.getName}] CONTENTS: $p"
}
