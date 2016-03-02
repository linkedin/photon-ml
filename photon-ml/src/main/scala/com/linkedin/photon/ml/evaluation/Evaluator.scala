package com.linkedin.photon.ml.evaluation

import org.apache.spark.rdd.RDD

/**
 * Interface for evaluation implementations
 *
 * @author xazhang
 */
trait Evaluator {

  /**
   * Evaluate scores
   *
   * @param score the scores to evaluate
   * @return score metric value
   */
  def evaluate(scores: RDD[(Long, Double)]): Double
}
