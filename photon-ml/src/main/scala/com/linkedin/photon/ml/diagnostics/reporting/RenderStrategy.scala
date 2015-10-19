package com.linkedin.photon.ml.diagnostics.reporting

/**
 * Defines a strategy for looking up renderers to facilitate writing generic renderers for composite types like
 * lists, chapters, etc. that can cleanly delegate as needed.
 *
 * @tparam R
 *           Rendered result type
 */
trait RenderStrategy[-P <: PhysicalReport, +R] {
  def locateRenderer(itemToRender:P):SpecificRenderer[P, R]
}
