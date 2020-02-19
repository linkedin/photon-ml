/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.data

import com.linkedin.photon.ml.Types.UniqueSampleId

/**
 * Local dataset implementation.
 *
 * @note One design concern is whether to store the local data as a [[Map]] or an [[Array]] (high sort cost, but low
 *       merge cost vs. no sort cost but high merge cost). Currently, we use an [[Array]] since the data is only sorted
 *       once, and used as the base for all other data/score [[Array]]s.
 *
 * @param dataPoints Local data points consists of (globalId, labeledPoint) pairs
 */
protected[ml] case class LocalDataset(dataPoints: Array[(UniqueSampleId, LabeledPoint)]) {

  require(
    dataPoints.length > 0,
    "Cannot create LocalDataset with empty data array")

  val numDataPoints: Int = dataPoints.length
  val numFeatures: Int = dataPoints
    .head
    ._2
    .features
    .length

  /**
   *
   * @return
   */
  def getLabels: Array[(UniqueSampleId, Double)] = dataPoints.map { case (uid, labeledPoint) =>
    (uid, labeledPoint.label)
  }

  /**
   *
   * @return
   */
  def getWeights: Array[(UniqueSampleId, Double)] = dataPoints.map { case (uid, labeledPoint) =>
    (uid, labeledPoint.weight)
  }

  /**
   *
   * @return
   */
  def getOffsets: Array[(UniqueSampleId, Double)] = dataPoints.map { case (uid, labeledPoint) =>
    (uid, labeledPoint.offset)
  }

  /**
   *
   * @return
   */
  def getUniqueIds: Array[UniqueSampleId] = dataPoints.map(_._1)

  /**
   * Add the residual scores to the offsets.
   *
   * @param residualScores The residual scores
   * @return The [[LocalDataset]] with updated offsets
   */
  def addScoresToOffsets(residualScores: Array[(UniqueSampleId, Double)]): LocalDataset = {

    val updatedDataPoints = dataPoints
      .zip(residualScores)
      .map { case ((dataId, LabeledPoint(label, features, offset, weight)), (residualScoreId, residualScoreDatum)) =>

        require(residualScoreId == dataId, s"residual score Id ($residualScoreId) and data Id ($dataId) don't match!")

        (dataId, LabeledPoint(label, features, residualScoreDatum + offset, weight))
      }

    LocalDataset(updatedDataPoints)
  }
}

object LocalDataset {

  /**
   * Factory method for LocalDataset.
   *
   * @param dataPoints The array of underlying data
   * @param isSortedByFirstIndex Whether or not to sort the data by global ID
   * @return A new LocalDataset
   */
  protected[ml] def apply(
      dataPoints: Array[(UniqueSampleId, LabeledPoint)],
      isSortedByFirstIndex: Boolean): LocalDataset = {

    if (isSortedByFirstIndex) {
      LocalDataset(dataPoints)
    } else {
      LocalDataset(dataPoints.sortBy(_._1))
    }
  }
}
