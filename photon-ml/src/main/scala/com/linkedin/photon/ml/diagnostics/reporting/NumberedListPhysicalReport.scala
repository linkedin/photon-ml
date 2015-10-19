package com.linkedin.photon.ml.diagnostics.reporting

/**
 * Represents a numbered list of items
 */
class NumberedListPhysicalReport(items:Seq[PhysicalReport]) extends SequencePhysicalReport[PhysicalReport](items) {
  override def toString():String = {
    s"NUMBERED LIST [ID: ${getId()}]"
  }
}
