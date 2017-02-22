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

class ChapterToStringRenderer(
    renderStrategy: RenderStrategy[SectionPhysicalReport, String],
    numberingContext: NumberingContext)
  extends SpecificRenderer[ChapterPhysicalReport, String] {

  private val baseRenderer =
    new BaseSequencePhysicalReportRender[SectionPhysicalReport, String](renderStrategy, numberingContext) {

    protected def coalesce(partialRendered: Seq[(List[Int], SectionPhysicalReport, String)]): String = {
      partialRendered.map(x => {
        val number = x._1.mkString(".")
        s"SECTION [$number] ${x._3}"
      }).mkString("\n")
    }
  }

  def render(c: ChapterPhysicalReport): String = {
    println(s"Rendering ${c.items.length} sections in this chapter")
    s"${c.title}\n${baseRenderer.render(c)}\n"
  }
}
