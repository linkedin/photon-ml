package com.linkedin.photon.ml.model

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.{KeyValueScore, GameData}

/**
 * Interface for the implementation of a GAME model
 *
 * @author xazhang
 */
trait Model {

  /**
   * Compute the score for the dataset
   *
   * @param dataPoints the dataset
   * @return the score
   */
  def score(dataPoints: RDD[(Long, GameData)]): KeyValueScore

  /**
   * Build a summary string for the model
   *
   * @return string representation
   */
  def toSummaryString: String
}
