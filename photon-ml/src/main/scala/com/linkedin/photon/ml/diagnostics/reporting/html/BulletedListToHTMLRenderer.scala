package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.mlease.spark.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender
import com.linkedin.photon.ml.diagnostics.reporting._

import scala.xml._

class BulletedListToHTMLRenderer(renderStrategy: RenderStrategy[PhysicalReport, Node],
                                 numberingContext: NumberingContext,
                                 namespaceBinding: NamespaceBinding,
                                 htmlPrefix: String,
                                 svgPrefix: String) extends SpecificRenderer[BulletedListPhysicalReport, Node] {

  private val baseRenderer = new BaseSequencePhysicalReportRender[PhysicalReport, Node](renderStrategy, numberingContext) {

    protected def coalesce(partiallyRendered: Seq[(List[Int], PhysicalReport, Node)]): Node = {
      new Group(partiallyRendered.map(x =>
        new Elem(htmlPrefix, "li", Null, namespaceBinding, true, x._3)))
    }
  }

  def render(lst: BulletedListPhysicalReport): Node = {
    val children = baseRenderer.render(lst)
    val list = new Elem(htmlPrefix, "ul", getAttributes(lst.getId), namespaceBinding, true, children)
    new Elem(htmlPrefix, "a", new PrefixedAttribute(htmlPrefix, "id", lst.getId.toString, Null), namespaceBinding, true, list)
  }

  private def getAttributes(id: Long): MetaData = {
    val idAttr = new PrefixedAttribute(htmlPrefix, "id", id.toString, Null)
    new PrefixedAttribute(htmlPrefix, "class", "bulleted_list", idAttr)
  }
}
