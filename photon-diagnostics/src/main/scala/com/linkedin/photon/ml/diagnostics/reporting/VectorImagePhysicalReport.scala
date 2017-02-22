/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.diagnostics.reporting

import scala.xml.Node

/**
 * @param caption
 * @param title
 */
abstract class VectorImagePhysicalReport(caption:Option[String] = None, title:Option[String] = None)
  extends GraphicPhysicalReport(caption, title) {

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
  def asSVG(height:Int=960, width:Int=1280):Node

  override def toString():String = {
    s"VECTOR IMAGE <- ${super.toString}"
  }
}
