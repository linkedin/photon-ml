package com.linkedin.photon.ml.diagnostics.hl

import com.google.common.base.Preconditions

/**
 * Model an individual histogram bin for purposes of HL tests. Probably want to think about several things:
 * <ul>
 * <li>Is there benefit from generalizing this to other payload types?</li>
 * <li>Is there benefit from generalizing this to other bin boundary types?</li>
 * </ul>
 *
 * For now, I won't worry too much about it
 *
 * @param lowerBound
 * The minimum predicted score for a sample to be inside this bin
 *
 * @param upperBound
 * The maximum predicted score for a sample to be inside this bin
 *
 * @param observedPosCount
 * How many positive samples have been observed in this bin
 *
 * @param observedNegCount
 * How many negative samples have been observed in this bin
 */
class PredictedProbabilityVersusObservedFrequencyHistogramBin(
    val lowerBound: Double,
    val upperBound: Double,
    var observedPosCount: Long = 0L,
    var observedNegCount: Long = 0L)
  extends Serializable {

  Preconditions.checkArgument(lowerBound >= 0 && lowerBound <= 1.0)
  Preconditions.checkArgument(upperBound >= 0 && upperBound <= 1.0)
  Preconditions.checkArgument(lowerBound < upperBound)
  Preconditions.checkArgument(observedNegCount >= 0L)
  Preconditions.checkArgument(observedPosCount >= 0L)

  def expectedPosCount(): Long = {
    Preconditions.checkArgument(observedNegCount >= 0L)
    Preconditions.checkArgument(observedPosCount >= 0L)

    val avgProbPos = (upperBound + lowerBound) / 2.0
    Preconditions.checkState(avgProbPos >= 0.0)
    Preconditions.checkState(avgProbPos <= 1.0)

    math.ceil((observedNegCount + observedPosCount) * avgProbPos).toLong
  }

  def expectedNegCount(): Long = {
    observedNegCount + observedPosCount - expectedPosCount
  }

  def accumulate(other: PredictedProbabilityVersusObservedFrequencyHistogramBin):
      PredictedProbabilityVersusObservedFrequencyHistogramBin = {
    observedNegCount += other.observedNegCount
    observedPosCount += other.observedPosCount
    this
  }

  override def toString(): String = {
    f"Range [$lowerBound%.012f, $upperBound%.012f) counts: [+/O $observedPosCount, +/E $expectedPosCount, " +
      f"-/O $observedNegCount, -/E $expectedNegCount]"
  }
}
