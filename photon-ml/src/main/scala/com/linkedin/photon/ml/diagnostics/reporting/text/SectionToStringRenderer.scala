package com.linkedin.photon.ml.diagnostics.reporting.text

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender

/**
 * Created by bdrew on 10/12/15.
 */
class SectionToStringRenderer(
    renderStrategy: RenderStrategy[PhysicalReport, String],
    numberingContext: NumberingContext)
  extends SpecificRenderer[SectionPhysicalReport, String] {

  private val baseRenderer =
    new BaseSequencePhysicalReportRender[PhysicalReport, String](renderStrategy, numberingContext) {

    protected def coalesce(partialRendered: Seq[(List[Int], PhysicalReport, String)]): String = {
      partialRendered.map(x => {
        val number = x._1.mkString(".")
        s"[$number] ${x._3}"
      }).mkString("\n")
    }
  }

  def render(s: SectionPhysicalReport): String = {
    println(s"Rendering ${s.items.length} items in this section")
    s"${s.title}\n${baseRenderer.render(s)}"
  }
}
