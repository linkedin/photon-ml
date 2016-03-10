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
class DefaultDownSampler(downSamplingRate: Double) extends DownSampler with Serializable {

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