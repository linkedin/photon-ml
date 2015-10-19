package com.linkedin.photon.ml.diagnostics.reporting

import java.awt.Image

/**
 * Intended for rasterized (i.e. non-vector) images
 */
class RasterizedImagePhysicalReport(val image:Image, caption:Option[String] = None, title:Option[String]=None) extends GraphicPhysicalReport(caption, title) {

  override def toString():String = {
    s"RASTERIZED IMAGE <- ${super.toString()}"
  }
}
