package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.photon.ml.diagnostics.reporting.{SpecificRenderer, PlotPhysicalReport, RenderStrategy, PhysicalReport, PlotUtils}
import scala.xml._

class PlotToHTMLRenderer(renderStrategy: RenderStrategy[PhysicalReport, Node],
                         namespaceBinding: NamespaceBinding,
                         htmlPrefix: String,
                         svgPrefix: String) extends SpecificRenderer[PlotPhysicalReport, Node] {
  def render(plot: PlotPhysicalReport): Node = {
    val svg = plot.asSVG()
    val comment = new Comment(plot.toString)

    val fig = plot.getCaption match {
      case Some(caption) =>
        val captionElt = new Elem(htmlPrefix, "figcaption", Null, namespaceBinding, true, new Text(caption))
        new Elem(htmlPrefix, "figure", getAttributes(plot.getId), namespaceBinding, true, captionElt, comment, svg)
      case None =>
        new Elem(htmlPrefix, "figure", getAttributes(plot.getId), namespaceBinding, true, comment, svg)
    }
    new Elem(htmlPrefix, "a", new PrefixedAttribute(htmlPrefix, "id", plot.getId.toString, Null), namespaceBinding, true, fig)
  }

  private def getAttributes(id: Long): MetaData = {
    val idAttr = new PrefixedAttribute(htmlPrefix, "id", id.toString, Null)

    // Add width and height constraints to force containing figure to be large enough to contain
    // the inner SVG diagram
    val styleAttr = new PrefixedAttribute(htmlPrefix, "style",
      s"min-width:${PlotUtils.PLOT_WIDTH}px; " +
      s"min-height:${PlotUtils.PLOT_HEIGHT}px", idAttr)

    new PrefixedAttribute(htmlPrefix, "class", "section_header", styleAttr)
  }
}
