package com.linkedin.photon.ml.sampler

import java.util.Random

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Interface for sampler implementations
 *
 * @author xazhang
 */
trait Sampler {

  /**
   * Downsample the dataset
   *
   * @param labeledPoints the dataset
   * @param seed random seed
   * @return downsampled dataset
   */
  def downSample(labeledPoints: RDD[(Long, LabeledPoint)], seed: Long = Sampler.getSeed): RDD[(Long, LabeledPoint)]
}

protected object Sampler {
  val random = new Random(MathConst.RANDOM_SEED)

  def getSeed: Long = random.nextLong()
}
