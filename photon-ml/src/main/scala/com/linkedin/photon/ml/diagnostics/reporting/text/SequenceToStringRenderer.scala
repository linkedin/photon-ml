package com.linkedin.photon.ml.diagnostics.reporting.text

import com.linkedin.photon.ml.diagnostics.reporting.PhysicalReport
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender
import com.linkedin.photon.ml.diagnostics.reporting.{RenderStrategy, NumberingContext, PhysicalReport}

/**
 * Generic rendering of sequences to text
 *
 * @param strategy
 * Instance of [[RenderStrategy]] that facilitates delegating the rendering of "atomic" things to an
 * appropriate renderer
 *
 * @param nc
 */
class SequenceToStringRenderer[-P <: PhysicalReport](strategy: RenderStrategy[P, String], nc: NumberingContext) extends BaseSequencePhysicalReportRender[P, String](strategy, nc) {
  protected def coalesce(partialRendered: Seq[(List[Int], P, String)]): String = {
    partialRendered.map(x => {
      val number = x._1.mkString(".")
      s"[$number] ${x._3}"
    }).mkString("\n")
  }
}
