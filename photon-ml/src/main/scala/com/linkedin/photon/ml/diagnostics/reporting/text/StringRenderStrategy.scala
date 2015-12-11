package com.linkedin.photon.ml.diagnostics.reporting.text

import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.base.RendererWrapper
import com.linkedin.photon.ml.diagnostics.reporting._

/**
 * Created by bdrew on 10/9/15.
 */
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
