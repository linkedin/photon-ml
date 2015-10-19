package com.linkedin.photon.ml.diagnostics.reporting

import scala.xml.Node

/**
 * @param caption
 * @param title
 */
abstract class VectorImagePhysicalReport(caption:Option[String] = None, title:Option[String] = None) extends GraphicPhysicalReport(caption, title) {
    /**
   * Generate a rasterized (i.e. rendered as pixels) version of this image.
   *
   * @param height
   *               Requested height in pixels
   * @param width
   *              Requested width in pixels
   * @param dpi
   *            Requested DPI for rasterization
   * @return
   *         Rasterized image
   */
  def asRasterizedImage(height:Int=960, width:Int=1280, dpi:Int=300):RasterizedImagePhysicalReport

  /**
   * Get the SVG corresponding to this image
   * @return
   */
  def asSVG():Node

  override def toString():String = {
    s"VECTOR IMAGE <- ${super.toString}"
  }
}
