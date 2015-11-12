/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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

import java.awt.Image

/**
 * Intended for rasterized (i.e. non-vector) images
 */
class RasterizedImagePhysicalReport(val image:Image, caption:Option[String] = None, title:Option[String]=None) extends GraphicPhysicalReport(caption, title) {

  override def toString():String = {
    s"RASTERIZED IMAGE <- ${super.toString()}"
  }
}
