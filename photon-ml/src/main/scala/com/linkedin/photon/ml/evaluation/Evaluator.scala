package com.linkedin.photon.ml.evaluation

import org.apache.spark.rdd.RDD

/**
 * @author xazhang
 */
trait Evaluator {
  def evaluate(scores: RDD[(Long, Double)]): Double
}
