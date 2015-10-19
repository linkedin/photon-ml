package com.linkedin.photon.ml.diagnostics.reporting.base

import com.linkedin.mlease.spark.diagnostics.reporting._
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
                                                                      strategy:RenderStrategy[P, R],
                                                                      numberingContext:NumberingContext) extends SpecificRenderer[SequencePhysicalReport[P], R] {
 
  def render(report:SequencePhysicalReport[P]) : R = {
    numberingContext.enterContext()
    try {
      coalesce(report.items.map( x => (numberingContext.nextItem(), x, strategy.locateRenderer(x).render(x))))
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
