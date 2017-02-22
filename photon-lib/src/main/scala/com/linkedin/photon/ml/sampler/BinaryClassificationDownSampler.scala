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

import scala.util.hashing.byteswap64

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Down-sampler implementation for binary classification problems. The positive instances are left as is. The negatives
 * are down-sampled as per the down-sampling rate and their weight is appropriately scaled.
 *
 * @param downSamplingRate The down sampling rate
 */
protected[ml] class BinaryClassificationDownSampler(downSamplingRate: Double) extends DownSampler with Serializable {
  require((downSamplingRate > 0D) && (downSamplingRate < 1D), s"Invalid down-sampling rate: $downSamplingRate")

  /**
   * Down-sample the negatives in the dataset.
   *
   * @param labeledPoints The dataset
   * @param seed Random seed
   * @return Down-sampled dataset
   */
  override def downSample(
      labeledPoints: RDD[(Long, LabeledPoint)],
      seed: Long = DownSampler.getSeed): RDD[(Long, LabeledPoint)] = {

    labeledPoints.mapPartitionsWithIndex(
      { (partitionIdx, iterator) =>
        val random = new Random(byteswap64(partitionIdx ^ seed))

        iterator.filter { case (_, labeledPoint) =>
            labeledPoint.label >= MathConst.POSITIVE_RESPONSE_THRESHOLD || random.nextDouble() < downSamplingRate
          }
          .map { case (id, labeledPoint) =>
            if (labeledPoint.label >= MathConst.POSITIVE_RESPONSE_THRESHOLD) {
              (id, labeledPoint)
            } else  {
              val updatedWeight = labeledPoint.weight / downSamplingRate
              (id, LabeledPoint(labeledPoint.label, labeledPoint.features, labeledPoint.offset, updatedWeight))
            }
          }
      },
      preservesPartitioning = true)
  }
}
