package com.linkedin.photon.ml.diagnostics.reporting

/**
 * A container representing a bulleted list of items.
 */
class BulletedListPhysicalReport(items:Seq[PhysicalReport]) extends SequencePhysicalReport[PhysicalReport](items) {
  override def toString():String = {
    s"BULLETED LIST [ID: ${getId()}]"
  }
}
