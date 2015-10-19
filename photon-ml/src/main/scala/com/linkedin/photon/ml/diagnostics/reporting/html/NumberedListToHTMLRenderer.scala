package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.mlease.spark.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender

import scala.xml._

class NumberedListToHTMLRenderer(renderStrategy: RenderStrategy[PhysicalReport, Node],
                                 numberingContext: NumberingContext,
                                 namespaceBinding: NamespaceBinding,
                                 htmlPrefix: String,
                                 svgPrefix: String) extends SpecificRenderer[NumberedListPhysicalReport, Node] {

  private val baseRenderer = new BaseSequencePhysicalReportRender[PhysicalReport, Node](renderStrategy, numberingContext) {

    protected def coalesce(partiallyRendered: Seq[(List[Int], PhysicalReport, Node)]): Node = {
      new Group(partiallyRendered.map(x =>
        new Elem(htmlPrefix, "li", Null, namespaceBinding, true, x._3)))
    }
  }

  def render(lst: NumberedListPhysicalReport): Node = {
    val children = baseRenderer.render(lst)
    val list = new Elem(htmlPrefix, "ol", getAttributes(lst.getId), namespaceBinding, true, children)
    new Elem(htmlPrefix, "a", new PrefixedAttribute(htmlPrefix, "id", lst.getId.toString, Null), namespaceBinding, true, list)
  }

  private def getAttributes(id: Long): MetaData = {
    val idAttr = new PrefixedAttribute(htmlPrefix, "id", id.toString, Null)
    new PrefixedAttribute(htmlPrefix, "class", "numbered_list", idAttr)
  }
}
