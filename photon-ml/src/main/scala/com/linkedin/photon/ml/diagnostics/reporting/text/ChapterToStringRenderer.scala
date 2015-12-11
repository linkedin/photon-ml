package com.linkedin.photon.ml.diagnostics.reporting.text

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender
import com.linkedin.photon.ml.diagnostics.reporting._

/**
 * Created by bdrew on 10/12/15.
 */
class ChapterToStringRenderer(renderStrategy: RenderStrategy[SectionPhysicalReport, String], numberingContext: NumberingContext) extends SpecificRenderer[ChapterPhysicalReport, String] {
  private val baseRenderer = new BaseSequencePhysicalReportRender[SectionPhysicalReport, String](renderStrategy, numberingContext) {
    protected def coalesce(partialRendered: Seq[(List[Int], SectionPhysicalReport, String)]): String = {
      partialRendered.map(x => {
        val number = x._1.mkString(".")
        s"SECTION [$number] ${x._3}"
      }).mkString("\n")
    }
  }

  def render(c: ChapterPhysicalReport): String = {
    println(s"Rendering ${c.items.length} sections in this chapter")
    s"${c.title}\n${baseRenderer.render(c)}\n"
  }
}
