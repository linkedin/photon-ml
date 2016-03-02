package com.linkedin.photon.ml.evaluation

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.util.Utils

/**
 * Evaluator for squared loss
 *
 * @param labelAndOffsetAndWeights label and offset weights
 * @param defaultScore default score
 * @author xazhang
 */
class SquaredLossEvaluator(
    labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))],
    defaultScore: Double = 0.0) extends Evaluator {

  /**
   * Evaluate scores
   *
   * @param score the scores to evaluate
   * @return score metric value
   */
  override def evaluate(scores: RDD[(Long, Double)]): Double = {
    val defaultScore = this.defaultScore

    val scoreAndLabelAndWeights = scores.rightOuterJoin(labelAndOffsetAndWeights)
        .mapValues { case (scoreOption, (label, offset, weight)) =>
      (scoreOption.getOrElse(defaultScore) + offset, (label, weight))
    }.values

    scoreAndLabelAndWeights.map { case (score, (label, weight)) =>
      val diff = score - label
      weight * diff * diff
    }.reduce(_ + _)
  }
}
