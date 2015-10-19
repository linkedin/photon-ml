package com.linkedin.photon.ml.diagnostics.reporting

trait SpecificRenderer[-P <: PhysicalReport, +R] {
  def render(thing:P):R
}
