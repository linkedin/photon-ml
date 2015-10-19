package com.linkedin.photon.ml.diagnostics.reporting

/**
 * Created by bdrew on 10/9/15.
 */
class SequencePhysicalReport[+Q <: PhysicalReport](val items:Seq[Q]) extends AbstractPhysicalReport {
}
