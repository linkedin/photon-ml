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
 * Default score binning approach.
 *
 * The strategy is very simple: pick what appears to be a sensible number of bins given the volume of data. This means that
 * sometimes, we will pick a number of bins that is too small (i.e. less than dimension + 1) because there isn't enough data
 * to support a finer binning strategy.
 */
class DefaultPredictedProbabilityVersusObservedFrequencyBinner extends AbstractPredictedProbabilityVersusObservedFrequencyBinner {

  import DefaultPredictedProbabilityVersusObservedFrequencyBinner._

  override def getBinCount(numItems: Long, numDimensions: Int): (String, Int) = {
    val msg: StringBuilder = new StringBuilder
    val desiredBinsBasedOnDimensions = estimateDesiredBinsFromDimension(numDimensions)
    val desiredBinsBasedOnData = estimateDesiredBinsFromData(numItems)
    val actualBins: Int = math.min(desiredBinsBasedOnData, desiredBinsBasedOnDimensions).toInt
    val okBinsMsg: String = if (actualBins >= desiredBinsBasedOnDimensions)
      "Sufficient bins for a discriminative test"
    else
      "Not enough bins for a discriminative test; please be careful when interpreting these results or rerun with more data"

    msg.append(s"Number of test set samples: $numItems\n")
      .append(s"Sample dimensionality: $numDimensions\n")
      .append(s"Target number of bins based on dimensionality alone: $desiredBinsBasedOnDimensions\n")
      .append(s"Target number of bins based on data alone: $desiredBinsBasedOnData\n")
      .append(okBinsMsg)
    (msg.toString, actualBins)
  }

  private def estimateDesiredBinsFromDimension(numDimensions: Long): Long = numDimensions + 2

  private def estimateDesiredBinsFromData(numItems: Long): Long = {
    // In principle, want to choose bins such that they are small enough to get good diagnostic power but large enough
    // so that it is very likely that they are all populated. This heuristic tries to trade those two quantities in
    // a reasonably data-driven way.
    (DATA_HEURISTIC_FACTOR_A * math.sqrt(numItems) + DATA_HEURISTIC_FACTOR_A * math.log1p(numItems)).toLong
  }
}

object DefaultPredictedProbabilityVersusObservedFrequencyBinner {
  val DATA_HEURISTIC_FACTOR_A = 0.9
  val DATA_HEURISTIC_FACTOR_B = 0.1
}
