/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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
