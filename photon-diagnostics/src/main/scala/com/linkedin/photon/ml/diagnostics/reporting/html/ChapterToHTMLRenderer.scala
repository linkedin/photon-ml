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
package com.linkedin.photon.ml.diagnostics.reporting.html

import scala.xml._

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender

class ChapterToHTMLRenderer(
    renderStrategy: RenderStrategy[SectionPhysicalReport, Node],
    numberingContext: NumberingContext,
    namespaceBinding: NamespaceBinding,
    htmlPrefix: String,
    svgPrefix: String)
  extends SpecificRenderer[ChapterPhysicalReport, Node] {

  private val baseRenderer =
    new BaseSequencePhysicalReportRender[SectionPhysicalReport, Node](renderStrategy, numberingContext) {

    protected def coalesce(partiallyRendered: Seq[(List[Int], SectionPhysicalReport, Node)]): Node = {
      new Group(partiallyRendered.map(x => x._3))
    }
  }

  def render(chapter: ChapterPhysicalReport): Node = {
    val children = baseRenderer.render(chapter)
    val anchor = new Elem(
      htmlPrefix, "a", new PrefixedAttribute(htmlPrefix, "id", chapter.getId.toString, Null),
      namespaceBinding, true, new Text(chapter.title))

    val heading = new Elem(htmlPrefix, "h1", Null, namespaceBinding, true, anchor)

    new Elem(htmlPrefix, "div", getAttributes(chapter.getId), namespaceBinding, true, heading, children)
  }

  private def getAttributes(id: Long): MetaData = {
    val idAttr = new PrefixedAttribute(htmlPrefix, "id", id.toString, Null)
    new PrefixedAttribute(htmlPrefix, "class", "chapter", idAttr)
  }
}
