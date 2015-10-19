package com.linkedin.photon.ml.stat

import com.linkedin.photon.ml.data
import com.linkedin.photon.ml.data.LabeledPoint
import org.apache.spark.mllib.linalg.VectorsWrapper
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD


/**
 * An object to generate basic statistics (e.g., max, min, mean, variance) of [[data.LabeledPoint]] RDD.
 *
 * @author dpeng
 */
private[ml] object BasicStatistics {

  /**
   * Generate basic statistics (e.g., max, min, mean, variance) of [[data.LabeledPoint]] RDD using the mllib
   * interface.
   * @param inputData Input data as [[data.LabeledPoint]] RDD
   * @return
   */
  def getBasicStatistics(inputData: RDD[LabeledPoint]): BasicStatisticalSummary = {
    val mllibSummary = Statistics.colStats(inputData.map(x => VectorsWrapper.breezeToMllib(x.features)))
    BasicStatisticalSummary(mllibSummary)
  }

}
