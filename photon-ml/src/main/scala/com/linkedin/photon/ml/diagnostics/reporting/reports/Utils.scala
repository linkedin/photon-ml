package com.linkedin.photon.ml.diagnostics.reporting.reports

import com.linkedin.photon.ml.io.GLMSuite


object Utils {
  def extractNameTerm(nt: String): (String, String) = {
    val split = nt.split(GLMSuite.DELIMITER)
    if (split.length > 1) {
      (split(0), split(1))
    } else {
      (nt, "")
    }
  }
}
