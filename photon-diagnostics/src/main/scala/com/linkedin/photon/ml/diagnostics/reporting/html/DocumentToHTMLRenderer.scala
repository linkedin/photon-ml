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

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender
import com.linkedin.photon.ml.diagnostics.reporting._

import scala.xml._

class DocumentToHTMLRenderer(
    renderStrategy: RenderStrategy[ChapterPhysicalReport, Node],
    numberingContext: NumberingContext,
    namespaceBinding: NamespaceBinding,
    htmlPrefix: String,
    svgPrefix: String)
  extends SpecificRenderer[DocumentPhysicalReport, Node] {

  private val baseRenderer =
    new BaseSequencePhysicalReportRender[ChapterPhysicalReport, Node](renderStrategy, numberingContext) {

    protected def coalesce(partiallyRendered: Seq[(List[Int], ChapterPhysicalReport, Node)]): Node = {
      new Group(partiallyRendered.map(x => x._3))
    }
  }

  def render(doc: DocumentPhysicalReport): Node = {
    val children = baseRenderer.render(doc)
    val style = new Elem(htmlPrefix, "style", Null, namespaceBinding, true, new Text(
      """
        |body {
        |  font-family: Arial, Helvetica, sans-serif;
        |  font-size: 12pt;
        |}
        |p {
        |  word-wrap: break-word;
        |}
        |h1 {
        |  font-size: 24pt;
        |  font-weight: bold;
        |}
        |h2 {
        |  font-size: 16pt;
        |  font-weight: bold;
        |}
      """.stripMargin))
    val header = new Elem(htmlPrefix, "head", Null, namespaceBinding, true,
      new Elem(htmlPrefix, "title", Null, namespaceBinding, true, new Text(doc.title)), style)
    val body = new Elem(htmlPrefix, "body", Null, namespaceBinding, true, children)
    new Elem(htmlPrefix, "html", Null, namespaceBinding, true, header, body)
  }

}
