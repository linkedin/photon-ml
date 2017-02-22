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
package com.linkedin.photon.ml.diagnostics.reporting.base

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting._

/**
 * Facilitates rendering potentially nested sequences of things
 *
 * @param strategy
 *                 Instance of [[RenderStrategy]] that facilitates delegating the rendering of "atomic" things to an
 *                 appropriate renderer
 *
 * @param numberingContext
 *                         State that handles tracking the numbering of items
 *
 * @tparam P
 *           Underlying physical report type
 *
 * @tparam R
 *           Output type
 */
abstract class BaseSequencePhysicalReportRender[-P <: PhysicalReport, R](
    strategy: RenderStrategy[P, R],
    numberingContext: NumberingContext)
  extends SpecificRenderer[SequencePhysicalReport[P], R] {

  def render(report:SequencePhysicalReport[P]) : R = {
    numberingContext.enterContext()
    try {
      coalesce(
        report.items
          .map( x => (numberingContext.nextItem(), x, strategy.locateRenderer(x).render(x))))
    } finally {
      numberingContext.exitContext()
    }
  }

  /**
   * Coalesce the per-item rendered results into a single result.
   *
   * E.g: if R is a string, concatenate all the partial item strings into a single one.
   *
   * @param partiallyRendered
   * @return
   */
  protected def coalesce(partiallyRendered: Seq[(List[Int], P, R)]): R
}
