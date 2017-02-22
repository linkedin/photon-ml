/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
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

import org.apache.spark.rdd.RDD

/**
 * Mock for injecting specific O/E histograms into Hosmer-Lemeshow diagnostic
 */
class MockPredictedProbabilityVersusObservedFrequencyBinner(val bins: Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]) extends PredictedProbabilityVersusObservedFrequencyBinner {
  def bin(numItems: Long, numDimensions: Int, observedVExpected: RDD[(Double, Double)]): (String, Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]) = {
    (MockPredictedProbabilityVersusObservedFrequencyBinner.BINNING_MESSAGE, bins)
  }
}

object MockPredictedProbabilityVersusObservedFrequencyBinner {

  val BINNING_MESSAGE = "Mock"
}
