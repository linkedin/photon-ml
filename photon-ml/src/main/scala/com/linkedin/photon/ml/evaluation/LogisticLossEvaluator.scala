package com.linkedin.photon.ml.evaluation

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.util.Utils


/**
 * Evaluator for logistic loss
 *
 * @param labelAndOffsetAndWeights label and offset weights
 * @param defaultScore default score
 * @author xazhang
 */
class LogisticLossEvaluator(
    labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))],
    defaultScore: Double = 0.0) extends Evaluator {

  /**
   * Evaluate scores
   *
   * @param score the scores to evaluate
   * @return score metric value
   */
  def evaluate(scores: RDD[(Long, Double)]): Double = {
    val defaultScore = this.defaultScore

    val scoreAndLabelAndWeights = scores.rightOuterJoin(labelAndOffsetAndWeights)
        .mapValues { case (scoreOption, (label, offset, weight)) =>
      (scoreOption.getOrElse(defaultScore) + offset, (label, weight))
    }.values

    scoreAndLabelAndWeights.map { case (score, (label, weight)) =>
      if (label > MathConst.POSITIVE_RESPONSE_THRESHOLD) {
        weight * Utils.log1pExp(-score)
      } else {
        weight * Utils.log1pExp(score)
      }
    }.reduce(_ + _)
  }
}
