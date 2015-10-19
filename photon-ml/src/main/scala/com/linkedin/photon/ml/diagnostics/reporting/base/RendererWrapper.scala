package com.linkedin.photon.ml.diagnostics.reporting.base

import com.linkedin.photon.ml.diagnostics.reporting.PhysicalReport
import com.linkedin.photon.ml.diagnostics.reporting.{Renderer, SpecificRenderer, PhysicalReport}

import scala.reflect.ClassTag

/**
 * Attempts to act as a type-safe wrapper around renderers for specific types, allowing them to masquerade as if they
 * were generic renderers.
 *
 * @param wrapped
 * The renderer to wrap
 */
class RendererWrapper[P <: PhysicalReport : ClassTag, +R](val wrapped: SpecificRenderer[P, R]) extends Renderer[R] {
  def render(p: PhysicalReport): R = {
    p match {
      case asP: P => wrapped.render(asP)
      case _ =>
        throw new ClassCastException(s"Wrapper could not cast instance of ${p.getClass.getName} to target type ${implicitly[ClassTag[P]].runtimeClass.getName}")
    }
  }
}
