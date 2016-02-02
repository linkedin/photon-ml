/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.stat

import breeze.linalg.Vector
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
   *
   */
  def getBasicStatistics(inputData: RDD[LabeledPoint]): BasicStatisticalSummary = {
    val mllibSummary = Statistics.colStats(inputData.map(x => VectorsWrapper.breezeToMllib(x.features)))
    val scale = if (mllibSummary.count > 0) { mllibSummary.count.toDouble } else { 1.0 }
    val meanAbs:Vector[Double] = VectorsWrapper.mllibToBreeze(mllibSummary.normL1) / scale
    BasicStatisticalSummary(mllibSummary, meanAbs)
  }

}
