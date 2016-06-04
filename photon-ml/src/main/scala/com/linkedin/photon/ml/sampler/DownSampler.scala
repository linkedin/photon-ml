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

import java.util.Random

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Interface for down-sampler implementations
  */
protected[ml] trait DownSampler {
  /**
    * Down-sample a dataset
    *
    * @param labeledPoints The dataset to down-sample
    * @param seed A random seed for down-sampling
    * @return The down-sampled dataset
    */
  def downSample(labeledPoints: RDD[(Long, LabeledPoint)], seed: Long = DownSampler.getSeed): RDD[(Long, LabeledPoint)]
}

object DownSampler {
  private val random = new Random(MathConst.RANDOM_SEED)

  protected[sampler] def getSeed: Long = random.nextLong()
}
