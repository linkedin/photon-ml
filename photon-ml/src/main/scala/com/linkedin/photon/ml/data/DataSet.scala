package com.linkedin.photon.ml.data


/**
 * Interface for GAME dataset implementations
 *
 * @author xazhang
 */
trait DataSet[D] {

  /**
   * Add scores to data offsets
   *
   * @param keyScore the scores
   */
  def addScoresToOffsets(keyScore: KeyValueScore): D

  /**
   * Build a summary string for the dataset
   *
   * @return string representation
   */
  def toSummaryString: String
}
