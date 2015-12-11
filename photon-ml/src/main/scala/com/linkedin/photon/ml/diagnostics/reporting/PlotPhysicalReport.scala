package com.linkedin.photon.ml.diagnostics.reporting

import java.awt.Graphics2D
import java.awt.Image
import java.awt.image.BufferedImage
import java.io.StringWriter
import javax.xml.parsers.DocumentBuilderFactory
import javax.xml.transform.stream.StreamResult
import javax.xml.transform.{OutputKeys, TransformerFactory}
import javax.xml.transform.dom.DOMSource
import javax.xml.transform.sax.SAXResult
import org.apache.batik.svggen.SVGGraphics2D

import com.xeiam.xchart.Chart

import scala.xml._
import scala.xml.parsing.NoBindingFactoryAdapter

/**
 * Represents a chart / plot
 *
 * @param plot
 *             The underlying chart / plot object
 *
 * @param caption
 *                Plot caption
 * @param title
 *              Plot title
 */
class PlotPhysicalReport(val plot:Chart, caption:Option[String] = None, title:Option[String] = None) extends VectorImagePhysicalReport(caption, title) {
  override def asRasterizedImage(height:Int=PlotUtils.PLOT_HEIGHT, width:Int=PlotUtils.PLOT_WIDTH, dpi:Int=300):RasterizedImagePhysicalReport = {
    val resultImage: Image = new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR)
    val graphics: Graphics2D = resultImage.getGraphics.asInstanceOf[Graphics2D]
    plot.paint(graphics, width, height)
    new RasterizedImagePhysicalReport(resultImage, caption, title)
  }

  override def asSVG(height:Int=PlotUtils.PLOT_HEIGHT, width:Int=PlotUtils.PLOT_WIDTH):Node = {
    val docFac = DocumentBuilderFactory.newInstance()
    val doc = docFac.newDocumentBuilder().newDocument()
    val ctx = new SVGGraphics2D(doc)
    try {
      plot.paint(ctx)
      var tmp = asXml(ctx.getRoot)

      new Elem(null, "svg", new PrefixedAttribute(tmp.namespace, "viewBox", s"0 0 $height $width",
        new UnprefixedAttribute("preserveAspectRatio", "xMidYMid meet",
          new UnprefixedAttribute("width", "100%",
            new UnprefixedAttribute("height", "100%", tmp.attributes)))), tmp.scope, true, tmp.child:_*)
    } finally {
      ctx.dispose()
    }
  }

  override def toString():String = {
    f"PLOT <- ${super.toString()}"
  }

  /**
   * Convert from old, Java DOM style APIs to Scala-style APIs
   * @param dom
   * @return
   */
  private def  asXml(dom: _root_.org.w3c.dom.Node): Node = {
    val source = new DOMSource(dom)
    val adapter = new NoBindingFactoryAdapter
    val saxResult = new SAXResult(adapter)
    val transformerFactory = javax.xml.transform.TransformerFactory.newInstance()
    val transformer = transformerFactory.newTransformer()
    transformer.transform(source, saxResult)
    adapter.rootElem
  }
}
