package com.linkedin.photon.ml.diagnostics.reporting

/**
 * A "simple" (i.e. contiguous chunk) of text
 * @param text
 */
class SimpleTextPhysicalReport(val text:String) extends AbstractPhysicalReport {
  override def toString():String = {
    s"TEXT [ID: ${getId}] $text"
  }
}
