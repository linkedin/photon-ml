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
import scala.util.hashing.byteswap64

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Down-sampler implementation for binary classification problems
 *
 * The positive instances are left as is. The negatives are down-sampled as per the down-sampling rate and their
 * weight is appropriate scaled
 *
 * @param downSamplingRate the down sampling rate
 *
 * @author xazhang
 * @author nkatariy
 */
class BinaryClassificationDownSampler(downSamplingRate: Double) extends DownSampler with Serializable {

  // TODO nkatariy We should have an assert on downsampling rate being > 0 and < 1 at runtime
  /**
   * Down-sample the negatives in the dataset
   *
   * @param labeledPoints the dataset
   * @param seed random seed
   * @return down-sampled dataset
   */
  override def downSample(labeledPoints: RDD[(Long, LabeledPoint)],
                          seed: Long = DownSampler.getSeed): RDD[(Long, LabeledPoint)] = {

    labeledPoints.mapPartitionsWithIndex({ case (partitionIdx, iterator) =>
      val random = new Random(byteswap64(partitionIdx ^ seed))

      iterator.filter { case (_, labeledPoint) =>
        labeledPoint.label >= MathConst.POSITIVE_RESPONSE_THRESHOLD || random.nextDouble() < downSamplingRate
      }.map { case (id, labeledPoint) =>
        if (labeledPoint.label >= MathConst.POSITIVE_RESPONSE_THRESHOLD) (id, labeledPoint)
        else  {
          val updatedWeight = labeledPoint.weight / downSamplingRate
          (id, LabeledPoint(labeledPoint.label, labeledPoint.features, labeledPoint.offset, updatedWeight))
        }
      }
    },
    preservesPartitioning = true)
  }
}