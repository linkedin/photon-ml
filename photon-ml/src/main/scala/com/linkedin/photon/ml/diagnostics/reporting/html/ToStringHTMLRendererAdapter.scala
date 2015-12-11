package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.photon.ml.diagnostics.reporting.PhysicalReport
import com.linkedin.photon.ml.diagnostics.reporting.{SimpleTextPhysicalReport, SpecificRenderer, PhysicalReport}

import scala.reflect.ClassTag
import scala.xml.{NamespaceBinding, Node}

/**
 * Created by bdrew on 10/12/15.
 */
class ToStringHTMLRendererAdapter[-P <: PhysicalReport : ClassTag](
    namespaceMap: NamespaceBinding,
    htmlPrefix: String,
    svgPrefix: String)
  extends SpecificRenderer[P, Node] {

  val wrappedRenderer = new SimpleTextToHTMLRenderer(namespaceMap, htmlPrefix, svgPrefix)

  def render(p: P): Node = {
    wrappedRenderer.render(
      new SimpleTextPhysicalReport(
        s"Do not know how to directly render instances of [${p.getClass.getName}] to HTML. Stringified version " +
        s"follows.\n$p"
      )
    )
  }

}
