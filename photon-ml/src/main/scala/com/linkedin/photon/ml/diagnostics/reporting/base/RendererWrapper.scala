/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
