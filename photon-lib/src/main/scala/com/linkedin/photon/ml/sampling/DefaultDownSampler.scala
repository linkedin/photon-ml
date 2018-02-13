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
package com.linkedin.photon.ml.sampling

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Default sampler implementation. This will act as a standard simple random sampler on the data set.
 * This should be used when all instances in the data set are equivalently important (e.g the labels are balanced).
 *
 * @param downSamplingRate The down sampling rate
 */
protected[ml] class DefaultDownSampler(override val downSamplingRate: Double) extends DownSampler with Serializable {

  /**
   * Down-sample the given data set.
   *
   * @param labeledPoints The full data set
   * @param seed A random seed
   * @return A down-sampled data set
   */
  override def downSample(
      labeledPoints: RDD[(UniqueSampleId, LabeledPoint)],
      seed: Long = DownSampler.getSeed): RDD[(UniqueSampleId, LabeledPoint)] =
    labeledPoints.sample(withReplacement = false, fraction = downSamplingRate, seed = seed)
}
