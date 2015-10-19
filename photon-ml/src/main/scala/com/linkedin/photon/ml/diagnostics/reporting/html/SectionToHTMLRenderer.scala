package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender
import com.linkedin.photon.ml.diagnostics.reporting._

import scala.xml._

/**
 * Render a section as HTML
 */
class SectionToHTMLRenderer(renderStrategy: RenderStrategy[PhysicalReport, Node],
                            numberingContext: NumberingContext,
                            namespaceBinding: NamespaceBinding,
                            htmlPrefix: String,
                            svgPrefix: String) extends SpecificRenderer[SectionPhysicalReport, Node] {

  private val baseRenderer = new BaseSequencePhysicalReportRender[PhysicalReport, Node](renderStrategy, numberingContext) {

    protected def coalesce(partiallyRendered: Seq[(List[Int], PhysicalReport, Node)]): Node = {
      new Group(partiallyRendered.map(x => x._3))
    }
  }

  def render(section: SectionPhysicalReport): Node = {
    val children = baseRenderer.render(section).asInstanceOf[Group]
    val anchor = new Elem(htmlPrefix, "a", new PrefixedAttribute(htmlPrefix, "id", section.getId.toString, Null), namespaceBinding, true, new Text(section.title))
    val heading = new Elem(htmlPrefix, "h2", Null, namespaceBinding, true, anchor)
    new Elem(htmlPrefix, "div", getAttributes(section.getId), namespaceBinding, true, heading, children)
  }

  private def getAttributes(id: Long): MetaData = {
    val idAttr = new PrefixedAttribute(htmlPrefix, "id", id.toString, Null)
    new PrefixedAttribute(htmlPrefix, "class", "section", idAttr)
  }
}
