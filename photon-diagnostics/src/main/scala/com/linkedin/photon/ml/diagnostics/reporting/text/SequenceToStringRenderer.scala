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

import com.linkedin.photon.ml.diagnostics.reporting.base.BaseSequencePhysicalReportRender
import com.linkedin.photon.ml.diagnostics.reporting.{NumberingContext, PhysicalReport, RenderStrategy}

/**
 * Generic rendering of sequences to text
 *
 * @param strategy
 * Instance of [[RenderStrategy]] that facilitates delegating the rendering of "atomic" things to an
 * appropriate renderer
 *
 * @param nc
 */
class SequenceToStringRenderer[-P <: PhysicalReport](strategy: RenderStrategy[P, String], nc: NumberingContext)
  extends BaseSequencePhysicalReportRender[P, String](strategy, nc) {

  protected def coalesce(partialRendered: Seq[(List[Int], P, String)]): String = {
    partialRendered.map(x => {
      val number = x._1.mkString(".")
      s"[$number] ${x._3}"
    }).mkString("\n")
  }
}
