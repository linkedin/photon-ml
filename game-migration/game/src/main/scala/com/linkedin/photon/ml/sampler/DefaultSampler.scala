package com.linkedin.photon.ml.sampler

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.contants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint


/**
 * @author xazhang
 */
class DefaultSampler(downSamplingRate: Double) extends Sampler with Serializable {

  val isDownSampling =
    if (math.abs(downSamplingRate - 1) < MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD) false
    else true

  override def downSample(labeledPoints: RDD[(Long, LabeledPoint)], seed: Long = Sampler.getSeed)
  : RDD[(Long, LabeledPoint)] = {

    labeledPoints.sample(withReplacement = false, fraction = downSamplingRate, seed = seed)
  }
}
