/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.diagnostics.hl

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

  require(lowerBound >= 0 && lowerBound <= 1.0)
  require(upperBound >= 0 && upperBound <= 1.0)
  require(lowerBound < upperBound)
  require(observedNegCount >= 0L)
  require(observedPosCount >= 0L)

  def expectedPosCount(): Long = {
    require(observedNegCount >= 0L)
    require(observedPosCount >= 0L)

    val avgProbPos = (upperBound + lowerBound) / 2.0
    require(avgProbPos >= 0.0)
    require(avgProbPos <= 1.0)

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
