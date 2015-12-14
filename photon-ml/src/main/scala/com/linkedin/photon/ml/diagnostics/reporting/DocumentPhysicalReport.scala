package com.linkedin.photon.ml.diagnostics.reporting

/**
 * Container for several chapters
 */
class DocumentPhysicalReport(chapters:Seq[ChapterPhysicalReport], val title:String)
  extends SequencePhysicalReport[ChapterPhysicalReport](chapters) {

}
