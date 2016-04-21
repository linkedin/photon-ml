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
package com.linkedin.photon.ml.sampler

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Default sampler implementation. This will act as a standard simple random sampler on the dataset.
 * This should be used when all instances in the dataset are equivalently important (e.g the labels are balanced)
 *
 * @param downSamplingRate the down sampling rate
 *
 * @author xazhang
 * @author nkatariy
 */
protected[ml] class DefaultDownSampler(downSamplingRate: Double) extends DownSampler with Serializable {

  // TODO nkatariy We should have an assert on downsampling rate being > 0 and < 1 at runtime
  /**
   * Samples from the given dataset
   *
   * @param labeledPoints the dataset
   * @param seed random seed
   * @return downsampled dataset
   */
  override def downSample(labeledPoints: RDD[(Long, LabeledPoint)],
                          seed: Long = DownSampler.getSeed): RDD[(Long, LabeledPoint)] = {
    labeledPoints.sample(withReplacement = false, fraction = downSamplingRate, seed = seed)
  }
}
