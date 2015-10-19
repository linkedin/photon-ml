package com.linkedin.photon.ml.diagnostics.reporting

/**
 * Common interface for things that are able to render physical reports into some final output form R.
 *
 * @tparam R
 *           Type of rendered result
 */
trait Renderer[+R] extends SpecificRenderer[PhysicalReport, R] {
}
