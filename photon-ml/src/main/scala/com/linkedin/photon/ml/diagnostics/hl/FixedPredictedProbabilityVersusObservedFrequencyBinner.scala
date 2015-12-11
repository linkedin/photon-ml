package com.linkedin.photon.ml.diagnostics.hl

import com.google.common.base.Preconditions

/**
 * Created by bdrew on 10/6/15.
 */
class FixedPredictedProbabilityVersusObservedFrequencyBinner(val numBins: Int)
  extends AbstractPredictedProbabilityVersusObservedFrequencyBinner {

  Preconditions.checkArgument(numBins > 0)

  def getBinCount(numItems: Long, numDimensions: Int): (String, Int) = ("Fixed number of bins", numBins)

}
