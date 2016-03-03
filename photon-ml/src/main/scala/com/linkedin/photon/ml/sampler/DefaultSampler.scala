package com.linkedin.photon.ml.sampler

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Default sampler implementation
 *
 * @param downSamplingRate the down sampling rate
 * @author xazhang
 */
class DefaultSampler(downSamplingRate: Double) extends Sampler with Serializable {

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

    labeledPoints.sample(withReplacement = false, fraction = downSamplingRate, seed = seed)
  }
}
