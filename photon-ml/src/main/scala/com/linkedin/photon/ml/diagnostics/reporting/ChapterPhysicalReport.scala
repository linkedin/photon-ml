package com.linkedin.photon.ml.diagnostics.reporting

/**
 * A container report representing a collection of sections
 *
 * @param sections
 */
class ChapterPhysicalReport(sections:Seq[SectionPhysicalReport], val title:String) extends SequencePhysicalReport[SectionPhysicalReport](sections) {
  override def toString():String = s"CHAPTER [ID: ${getId()}, TITLE:$title]"
}
