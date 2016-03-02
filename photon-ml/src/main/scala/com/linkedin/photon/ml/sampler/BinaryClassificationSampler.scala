package com.linkedin.photon.ml.sampler

import java.util.Random

import org.apache.spark.rdd.RDD
import scala.util.hashing.byteswap64

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Sampler implementation for binary classification problems
 *
 * @param downSamplingRate the down sampling rate
 * @author xazhang
 */
class BinaryClassificationSampler(downSamplingRate: Double) extends Sampler with Serializable {

  val isDownSampling = math.abs(downSamplingRate - 1) < MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD

  /**
   * Downsample the dataset
   *
   * @param labeledPoints the dataset
   * @param seed random seed
   * @return downsampled dataset
   */
  override def downSample(
      labeledPoints: RDD[(Long, LabeledPoint)], seed: Long = Sampler.getSeed): RDD[(Long, LabeledPoint)] = {

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
