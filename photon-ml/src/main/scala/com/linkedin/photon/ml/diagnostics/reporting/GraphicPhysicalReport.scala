package com.linkedin.photon.ml.diagnostics.reporting

/**
 * Describes some kind of graphic to be reported
 *
 * @param caption
 *
 * @param title
 *
 */
abstract class GraphicPhysicalReport(caption:Option[String] = None, title:Option[String] = None) extends AbstractPhysicalReport {
  def getCaption():Option[String] = caption
  def getTitle():Option[String] = title
  override def toString():String = {
    s"GRAPHIC [ID: ${getId()}, CAPTION: ${getCaption()}, TITLE: ${getTitle()}"
  }
}
