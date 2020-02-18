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

import java.util.Random

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Interface for down-sampler implementations.
 */
protected[ml] trait DownSampler {

  import DownSampler._

  // The down-sampling rate
  val downSamplingRate: Double

  // Reject invalid down-sampling rates
  require(isValidDownSamplingRate(downSamplingRate), s"Invalid down-sampling rate: $downSamplingRate")

  /**
   * Down-sample a dataset.
   *
   * @param labeledPoints The dataset to down-sample
   * @param seed A random seed for down-sampling
   * @return The down-sampled dataset
   */
  def downSample(
      labeledPoints: RDD[LabeledPoint],
      seed: Long = getSeed): RDD[LabeledPoint]
}

object DownSampler {

  private val random = new Random(MathConst.RANDOM_SEED)

  /**
   * Get a random seed for down-sampling.
   *
   * @return A random Long
   */
  protected[sampling] def getSeed: Long = random.nextLong()

  /**
   * Check if a given down-sampling rate is valid.
   *
   * @param downSamplingRate The down-sampling rate
   * @return True if given a valid down-sampling rate, false otherwise
   */
  def isValidDownSamplingRate(downSamplingRate: Double): Boolean = (downSamplingRate < 1D) && (downSamplingRate > 0D)
}
