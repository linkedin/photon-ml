package com.linkedin.photon.ml.diagnostics.reporting

/**
 * Simple composite that defines a sequence of child reports that, collectively, represent one section of a report.
 *
 * @param contents
 * @param title
 */
class SectionPhysicalReport(contents:Seq[PhysicalReport], val title:String) extends SequencePhysicalReport[PhysicalReport](contents) {
  override def toString():String = {
    s"SECTION [ID: ${getId} TITLE: $title]"
  }
}
