package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.photon.ml.diagnostics.reporting.SimpleTextPhysicalReport
import com.linkedin.photon.ml.diagnostics.reporting.{SimpleTextPhysicalReport, SpecificRenderer}

import scala.xml._

/**
 * Render simple text to a sequence of &lt;p/&gt; tags
 */
class SimpleTextToHTMLRenderer(namespaceBinding: NamespaceBinding, htmlPrefix: String, svgPrefix: String)
  extends SpecificRenderer[SimpleTextPhysicalReport, Node] {

  def render(text: SimpleTextPhysicalReport): Node = {
    val paragraphs = text.text.split("\\n+").map(contents =>
      new Elem(htmlPrefix, "p", getAttributes(text.getId), namespaceBinding, true, new Text(contents)))
    new Elem(
      htmlPrefix, "a", new PrefixedAttribute(htmlPrefix, "id", text.getId.toString, Null),
      namespaceBinding, true, paragraphs: _*)
  }

  private def getAttributes(id: Long): MetaData = {
    new PrefixedAttribute(htmlPrefix, "class", "simple_text", Null)
  }
}
