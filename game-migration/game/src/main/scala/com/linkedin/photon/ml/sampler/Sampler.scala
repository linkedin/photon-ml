package com.linkedin.photon.ml.sampler

import java.util.Random

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.contants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint


/**
 * @author xazhang
 */
trait Sampler {
  def downSample(labeledPoints: RDD[(Long, LabeledPoint)], seed: Long = Sampler.getSeed): RDD[(Long, LabeledPoint)]
}

protected object Sampler {
  val random = new Random(MathConst.RANDOM_SEED)

  def getSeed: Long = random.nextLong()
}
