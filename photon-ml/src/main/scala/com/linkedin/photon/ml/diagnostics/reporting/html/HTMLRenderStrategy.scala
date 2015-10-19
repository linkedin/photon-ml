package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.RendererWrapper

import scala.xml._


class HTMLRenderStrategy(val htmlPrefix: String = null, val svgPrefix: String = null, val namespaceMap: NamespaceBinding = TopScope) extends RenderStrategy[PhysicalReport, Node] {
  val chapterSectionNumbering = new NumberingContext()
  val listNumbering = new NumberingContext()

  def locateRenderer(p: PhysicalReport): Renderer[Node] = {
    p match {
      case _: BulletedListPhysicalReport =>
        new RendererWrapper[BulletedListPhysicalReport, Node](new BulletedListToHTMLRenderer(this, listNumbering, namespaceMap, htmlPrefix, svgPrefix))
      case _: ChapterPhysicalReport =>
        new RendererWrapper[ChapterPhysicalReport, Node](new ChapterToHTMLRenderer(this, chapterSectionNumbering, namespaceMap, htmlPrefix, svgPrefix))
      case _: DocumentPhysicalReport =>
        new RendererWrapper[DocumentPhysicalReport, Node](new DocumentToHTMLRenderer(this, chapterSectionNumbering, namespaceMap, htmlPrefix, svgPrefix))
      case _: NumberedListPhysicalReport =>
        new RendererWrapper[NumberedListPhysicalReport, Node](new NumberedListToHTMLRenderer(this, listNumbering, namespaceMap, htmlPrefix, svgPrefix))
      case _: PlotPhysicalReport =>
        new RendererWrapper[PlotPhysicalReport, Node](new PlotToHTMLRenderer(this, namespaceMap, htmlPrefix, svgPrefix))
      case _: SectionPhysicalReport =>
        new RendererWrapper[SectionPhysicalReport, Node](new SectionToHTMLRenderer(this, chapterSectionNumbering, namespaceMap, htmlPrefix, svgPrefix))
      case _: SimpleTextPhysicalReport =>
        new RendererWrapper[SimpleTextPhysicalReport, Node](new SimpleTextToHTMLRenderer(namespaceMap, htmlPrefix, svgPrefix))
      case _: ReferencePhysicalReport =>
        new RendererWrapper[ReferencePhysicalReport, Node](new ReferenceToHTMLRenderer(namespaceMap, htmlPrefix, svgPrefix))
      case _ =>
        new RendererWrapper[PhysicalReport, Node](new ToStringHTMLRendererAdapter[PhysicalReport](namespaceMap, htmlPrefix, svgPrefix))
    }
  }

}
