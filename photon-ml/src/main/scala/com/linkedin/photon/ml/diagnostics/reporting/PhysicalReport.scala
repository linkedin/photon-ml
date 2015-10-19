package com.linkedin.photon.ml.diagnostics.reporting

/**
 * A physical report is the result of converting a logical report into a presentable form. While the logical reports
 * are focused on semantics and content, this is focused on things like representing "here's a chunk of text and a
 * picture that I want to show up together somewhere"
 */
trait PhysicalReport {
  def getId():Long
}
