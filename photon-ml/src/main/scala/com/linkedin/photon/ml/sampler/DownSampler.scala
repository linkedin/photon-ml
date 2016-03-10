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

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Interface for down-sampler implementations
 *
 * @author xazhang
 * @author nkatariy
 */
trait DownSampler {

  /**
   * Down-sample the dataset
   *
   * @param labeledPoints the dataset
   * @param seed random seed
   * @return down-sampled dataset
   */
  def downSample(labeledPoints: RDD[(Long, LabeledPoint)], seed: Long = DownSampler.getSeed): RDD[(Long, LabeledPoint)]
}

protected object DownSampler {
  val random = new Random(MathConst.RANDOM_SEED)

  def getSeed: Long = random.nextLong()
}