/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
 * Generic interface for binning logistic-regression type responses into a list of (prediction range -> actual counts v expected counts) tuples
 * It is possible, in the future, that it would make sense to further refine / abstract this class if we think about doing general &Chi;<sup>2</sup>
 * testing or similar goodness-of-fit measures.
 *
 * For now, the major purpose of this class is to allow me to mock this logic out for testing.
 */
trait PredictedProbabilityVersusObservedFrequencyBinner {
  def bin(numItems: Long, numDimensions: Int, observedVExpected: RDD[(Double, Double)]): (String, Array[PredictedProbabilityVersusObservedFrequencyHistogramBin])
}
