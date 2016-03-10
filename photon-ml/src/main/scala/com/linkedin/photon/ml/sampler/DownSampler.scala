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