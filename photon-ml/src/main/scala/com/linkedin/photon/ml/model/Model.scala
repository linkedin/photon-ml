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
   * Compute the score for the GAME data set.
   *
   * @param dataPoints the dataset, which is a RDD consists of the (global Id, GameData) pairs. Note that the Long in
   *                   the RDD above is a unique identifier for which GenericRecord the GameData object was created,
   *                   referred to in the GAME code as the "global ID".
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
