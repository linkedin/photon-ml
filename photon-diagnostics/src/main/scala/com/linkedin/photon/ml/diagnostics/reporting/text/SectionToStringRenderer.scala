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
package com.linkedin.photon.ml.diagnostics.reporting.text

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender

class SectionToStringRenderer(
    renderStrategy: RenderStrategy[PhysicalReport, String],
    numberingContext: NumberingContext)
  extends SpecificRenderer[SectionPhysicalReport, String] {

  private val baseRenderer =
    new BaseSequencePhysicalReportRender[PhysicalReport, String](renderStrategy, numberingContext) {

    protected def coalesce(partialRendered: Seq[(List[Int], PhysicalReport, String)]): String = {
      partialRendered.map(x => {
        val number = x._1.mkString(".")
        s"[$number] ${x._3}"
      }).mkString("\n")
    }
  }

  def render(s: SectionPhysicalReport): String = {
    println(s"Rendering ${s.items.length} items in this section")
    s"${s.title}\n${baseRenderer.render(s)}"
  }
}
