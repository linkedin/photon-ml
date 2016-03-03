package com.linkedin.photon.ml.evaluation

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

/**
 * Evaluator for binary classification problems
 *
 * @param labelAndOffsets sequence of labels and offsets
 * @param defaultScore default score
 * @author xazhang
 */
class BinaryClassificationEvaluator(labelAndOffsets: RDD[(Long, (Double, Double))], defaultScore: Double = 0.0)
  extends Evaluator {

  /**
   * Evaluate scores
   *
   * @param score the scores to evaluate
   * @return score metric value
   */
  override def evaluate(scores: RDD[(Long, Double)]): Double = {
    // Create a local copy of the defaultScore, so that the underlying object won't get shipped to the executor nodes
    val defaultScore = this.defaultScore
    val scoreAndLabels = scores.rightOuterJoin(labelAndOffsets).mapValues { case (scoreOption, (label, offset)) =>
      (scoreOption.getOrElse(defaultScore) + offset, label)
    }.values

    new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
  }
}
