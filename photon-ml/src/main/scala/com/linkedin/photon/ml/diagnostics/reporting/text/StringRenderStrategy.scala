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
package com.linkedin.photon.ml.diagnostics.reporting.text

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.RendererWrapper
import com.linkedin.photon.ml.diagnostics.reporting._


class StringRenderStrategy extends RenderStrategy[PhysicalReport, String] {
  val chapterSectionNumberingContext = new NumberingContext()
  val listNumberingContext = new NumberingContext()

  def locateRenderer(itemToRender: PhysicalReport): SpecificRenderer[PhysicalReport, String] = {
    itemToRender match {
      case _: DocumentPhysicalReport => new RendererWrapper[DocumentPhysicalReport, String](
        new DocumentToStringRenderer(this, chapterSectionNumberingContext))

      case _: ChapterPhysicalReport => new RendererWrapper[ChapterPhysicalReport, String](
        new ChapterToStringRenderer(this, chapterSectionNumberingContext))

      case _: SectionPhysicalReport => new RendererWrapper[SectionPhysicalReport, String](
        new SectionToStringRenderer(this, chapterSectionNumberingContext))

      case _: BulletedListPhysicalReport => new RendererWrapper[BulletedListPhysicalReport, String](
        new SequenceToStringRenderer(this, listNumberingContext))

      case _: NumberedListPhysicalReport => new RendererWrapper[NumberedListPhysicalReport, String](
        new SequenceToStringRenderer(this, listNumberingContext))

      case _: PhysicalReport => new RendererWrapper[PhysicalReport, String](new AnyToStringRenderer[PhysicalReport]())
    }
  }
}
