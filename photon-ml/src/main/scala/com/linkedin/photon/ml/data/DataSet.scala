package com.linkedin.photon.ml.data


/**
 * @author xazhang
 */
trait DataSet[D] {
  def addScoresToOffsets(keyScore: KeyValueScore): D

  def toSummaryString: String
}
