package com.linkedin.photon.ml.diagnostics.reporting.text

import com.linkedin.mlease.spark.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender

/**
 * Created by bdrew on 10/12/15.
 */
class DocumentToStringRenderer(renderStrategy: RenderStrategy[ChapterPhysicalReport, String], numberingContext: NumberingContext) extends SpecificRenderer[DocumentPhysicalReport, String] {
  private val baseRenderer = new BaseSequencePhysicalReportRender[ChapterPhysicalReport, String](renderStrategy, numberingContext) {
    protected def coalesce(partialRendered: Seq[(List[Int], ChapterPhysicalReport, String)]): String = {
      partialRendered.map(x => {
        val number = x._1.mkString(".")
        s"CHAPTER [$number] ${x._3}"
      }).mkString("\n")
    }
  }

  def render(c: DocumentPhysicalReport): String = {
    println(s"Rendering ${c.items.length} chapters in this document")
    s"${c.title}\n${baseRenderer.render(c)}\n"
  }

}
