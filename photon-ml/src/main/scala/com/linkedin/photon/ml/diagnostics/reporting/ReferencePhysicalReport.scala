package com.linkedin.photon.ml.diagnostics.reporting

/**
 * Represents a reference to another item
 * @param referee
 *                The object to which a reference should be generated
 */
class ReferencePhysicalReport(val referee:PhysicalReport, val msg:String) extends AbstractPhysicalReport {
  override def toString():String = {
    s"REFERENCE [ID: {$getId} REFEREE ID: ${referee.getId} TEXT: $msg]"
  }
}
