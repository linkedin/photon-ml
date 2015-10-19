package com.linkedin.photon.ml.function

/**
 * A trait to represent whether an L1 regularization is applied to a [[DiffFunction]] or [[TwiceDiffFunction]].
 */
trait L1RegularizationTerm {
  def getL1RegularizationParam: Double
}
