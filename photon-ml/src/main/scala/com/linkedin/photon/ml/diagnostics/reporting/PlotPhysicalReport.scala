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

import scala.xml.{XML, Node}
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
  override def asRasterizedImage(height:Int=960, width:Int=1280, dpi:Int=300):RasterizedImagePhysicalReport = {
    val resultImage: Image = new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR)
    val graphics: Graphics2D = resultImage.getGraphics.asInstanceOf[Graphics2D]
    plot.paint(graphics, width, height)
    new RasterizedImagePhysicalReport(resultImage, caption, title)
  }

  override def asSVG():Node = {
    val docFac = DocumentBuilderFactory.newInstance()
    val doc = docFac.newDocumentBuilder().newDocument()
    val ctx = new SVGGraphics2D(doc)
    try {
      plot.paint(ctx)
      asXml(ctx.getRoot)
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
//    val transformer = TransformerFactory.newInstance().newTransformer()
//    transformer.setOutputProperty(OutputKeys.ENCODING, "UTF-8")
//    transformer.setOutputProperty(OutputKeys.INDENT, "yes")
//    val buffer = new StringWriter()
//    transformer.transform(new DOMSource(dom), new StreamResult(buffer))
//    XML.loadString(buffer.getBuffer.toString)

    val source = new DOMSource(dom)
    val adapter = new NoBindingFactoryAdapter
    val saxResult = new SAXResult(adapter)
    val transformerFactory = javax.xml.transform.TransformerFactory.newInstance()
    val transformer = transformerFactory.newTransformer()
    transformer.transform(source, saxResult)
    adapter.rootElem
  }
}
