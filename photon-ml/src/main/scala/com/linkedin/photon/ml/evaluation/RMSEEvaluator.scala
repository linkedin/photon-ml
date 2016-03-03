package com.linkedin.photon.ml.evaluation

import org.apache.spark.rdd.RDD


/**
 * Evaluator for root mean squared error
 *
 * @param labelAndOffsetAndWeights label and offset weights
 * @param defaultScore default score
 * @author xazhang
 */
class RMSEEvaluator(
    labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))],
    defaultScore: Double = 0.0) extends Evaluator {

  val squaredLossEvaluator = new SquaredLossEvaluator(labelAndOffsetAndWeights, defaultScore)

  /**
   * Evaluate scores
   *
   * @param score the scores to evaluate
   * @return score metric value
   */
  override def evaluate(scores: RDD[(Long, Double)]): Double = {
    math.sqrt(squaredLossEvaluator.evaluate(scores) / labelAndOffsetAndWeights.count())
  }
}
