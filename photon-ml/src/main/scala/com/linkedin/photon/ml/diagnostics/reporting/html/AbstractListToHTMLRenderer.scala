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

import scala.xml._

class AbstractListToHTMLRenderer[-L <: SequencePhysicalReport[PhysicalReport]](listTag:String,
                                                                              renderStrategy: RenderStrategy[PhysicalReport, Node],
                                                                              numberingContext: NumberingContext,
                                                                              namespaceBinding: NamespaceBinding,
                                                                              htmlPrefix: String,
                                                                              svgPrefix: String) extends SpecificRenderer[L, Node] {

  private val baseRenderer = new BaseSequencePhysicalReportRender[PhysicalReport, Node](renderStrategy, numberingContext) {

    protected def coalesce(partiallyRendered: Seq[(List[Int], PhysicalReport, Node)]): Node = {
      new Group(partiallyRendered.map(x =>
        x._2 match {
          case st:SimpleTextPhysicalReport =>
            new Elem(htmlPrefix, "li", Null, namespaceBinding, true, new Text(st.text))
          case nl:NumberedListPhysicalReport =>
            x._3
          case bl:BulletedListPhysicalReport =>
            x._3
          case other:PhysicalReport =>
            new Elem(htmlPrefix, "li", Null, namespaceBinding, true, x._3)
        }))
    }
  }

  def render(lst: L): Node = {
    val children = baseRenderer.render(lst)
    val list = new Elem(htmlPrefix, listTag, getAttributes(lst.getId), namespaceBinding, true, children)
    new Elem(htmlPrefix, "a", new PrefixedAttribute(htmlPrefix, "id", lst.getId.toString, Null), namespaceBinding, true, list)
  }

  private def getAttributes(id: Long): MetaData = {
    val idAttr = new PrefixedAttribute(htmlPrefix, "id", id.toString, Null)
    new PrefixedAttribute(htmlPrefix, "class", "numbered_list", idAttr)
  }
}
