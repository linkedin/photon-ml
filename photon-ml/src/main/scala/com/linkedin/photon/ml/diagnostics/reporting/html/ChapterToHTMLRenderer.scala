package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender
import com.linkedin.photon.ml.diagnostics.reporting._

import scala.xml._

class ChapterToHTMLRenderer(
    renderStrategy: RenderStrategy[SectionPhysicalReport, Node],
    numberingContext: NumberingContext,
    namespaceBinding: NamespaceBinding,
    htmlPrefix: String,
    svgPrefix: String)
  extends SpecificRenderer[ChapterPhysicalReport, Node] {

  private val baseRenderer =
    new BaseSequencePhysicalReportRender[SectionPhysicalReport, Node](renderStrategy, numberingContext) {

    protected def coalesce(partiallyRendered: Seq[(List[Int], SectionPhysicalReport, Node)]): Node = {
      new Group(partiallyRendered.map(x => x._3))
    }
  }

  def render(chapter: ChapterPhysicalReport): Node = {
    val children = baseRenderer.render(chapter)
    val anchor = new Elem(
      htmlPrefix, "a", new PrefixedAttribute(htmlPrefix, "id", chapter.getId.toString, Null),
      namespaceBinding, true, new Text(chapter.title))

    val heading = new Elem(htmlPrefix, "h1", Null, namespaceBinding, true, anchor)

    new Elem(htmlPrefix, "div", getAttributes(chapter.getId), namespaceBinding, true, heading, children)
  }

  private def getAttributes(id: Long): MetaData = {
    val idAttr = new PrefixedAttribute(htmlPrefix, "id", id.toString, Null)
    new PrefixedAttribute(htmlPrefix, "class", "chapter", idAttr)
  }
}
