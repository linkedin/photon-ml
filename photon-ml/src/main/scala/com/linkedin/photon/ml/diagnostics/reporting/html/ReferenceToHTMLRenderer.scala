package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.photon.ml.diagnostics.reporting.PhysicalReport
import com.linkedin.photon.ml.diagnostics.reporting.{ReferencePhysicalReport, SpecificRenderer}

import scala.xml._


class ReferenceToHTMLRenderer(
    namespaceBinding: NamespaceBinding,
    htmlPrefix: String,
    svgPrefix: String)
  extends SpecificRenderer[ReferencePhysicalReport, Node] {

  def render(reference: ReferencePhysicalReport): Node = {
    new Elem(
      htmlPrefix, "a", new PrefixedAttribute(htmlPrefix, "href", "#" + reference.referee.getId.toString, Null),
      namespaceBinding, true, new Text(reference.msg))
  }
}
