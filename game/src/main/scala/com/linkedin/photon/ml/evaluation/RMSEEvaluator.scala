package com.linkedin.photon.ml.evaluation

import org.apache.spark.rdd.RDD


/**
 * @author xazhang
 */
class RMSEEvaluator(
    labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))],
    defaultScore: Double = 0.0) extends Evaluator {

  val squaredLossEvaluator = new SquaredLossEvaluator(labelAndOffsetAndWeights, defaultScore)

  override def evaluate(scores: RDD[(Long, Double)]): Double = {
    math.sqrt(squaredLossEvaluator.evaluate(scores) / labelAndOffsetAndWeights.count())
  }
}
