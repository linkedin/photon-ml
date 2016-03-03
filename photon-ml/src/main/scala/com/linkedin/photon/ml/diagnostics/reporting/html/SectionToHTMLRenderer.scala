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

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender
import com.linkedin.photon.ml.diagnostics.reporting._

import scala.xml._

/**
 * Render a section as HTML
 */
class SectionToHTMLRenderer(renderStrategy: RenderStrategy[PhysicalReport, Node],
                            numberingContext: NumberingContext,
                            namespaceBinding: NamespaceBinding,
                            htmlPrefix: String,
                            svgPrefix: String) extends SpecificRenderer[SectionPhysicalReport, Node] {

  private val baseRenderer =
    new BaseSequencePhysicalReportRender[PhysicalReport, Node](renderStrategy, numberingContext) {

    protected def coalesce(partiallyRendered: Seq[(List[Int], PhysicalReport, Node)]): Node = {
      new Group(partiallyRendered.map(x => x._3))
    }
  }

  def render(section: SectionPhysicalReport): Node = {
    val children = baseRenderer.render(section).asInstanceOf[Group]
    val anchor = new Elem(
      htmlPrefix, "a", new PrefixedAttribute(htmlPrefix, "id", section.getId.toString, Null), namespaceBinding,
      true, new Text(section.title))
    val heading = new Elem(htmlPrefix, "h2", Null, namespaceBinding, true, anchor)
    new Elem(htmlPrefix, "div", getAttributes(section.getId), namespaceBinding, true, heading, children)
  }

  private def getAttributes(id: Long): MetaData = {
    val idAttr = new PrefixedAttribute(htmlPrefix, "id", id.toString, Null)
    new PrefixedAttribute(htmlPrefix, "class", "section", idAttr)
  }
}
