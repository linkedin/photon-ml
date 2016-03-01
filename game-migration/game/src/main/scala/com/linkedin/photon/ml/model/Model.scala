package com.linkedin.photon.ml.model

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.{KeyValueScore, GameData}


/**
 * @author xazhang
 */
trait Model {

  def score(dataPoints: RDD[(Long, GameData)]): KeyValueScore

  def toSummaryString: String
}
