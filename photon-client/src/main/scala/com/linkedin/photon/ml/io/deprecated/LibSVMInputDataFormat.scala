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
package com.linkedin.photon.ml.io.deprecated

import breeze.linalg.SparseVector
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.index.{IdentityIndexMapLoader, IndexMapLoader}

/**
 * An input format supporting reading LibSVM txt, the assumed format is:
 * [label] [feature_index]:[feature_value] [feature_index]:[feature_value] ....           (delimited by space)
 *
 * By default, libsvm feature indices start from 1, but if zeroBased is true, we will assume it
 * starts from 0.
 */
class LibSVMInputDataFormat(
    val featureDimension: Int,
    val useIntercept: Boolean = true,
    val zeroBased: Boolean = false,
    val delim: String = " ",
    val idxValueDelim: String = ":"
  ) extends InputDataFormat {

  private val trueFeatureDimension: Int = if (useIntercept) featureDimension + 1 else featureDimension

  private val _indexMapLoader = new IdentityIndexMapLoader(trueFeatureDimension, useIntercept)

  /**
   *
   * @param sc The spark context
   * @param inputPath Input path of labeled points
   * @param selectedFeaturesPath Optional path of selected features
   * @param minPartitions Minimum number of partitions
   * @return An RDD of LabeledPoints
   */
  override def loadLabeledPoints(
      sc: SparkContext,
      inputPath: String,
      selectedFeaturesPath: Option[String],
      minPartitions: Int): RDD[LabeledPoint] = {
    val itemDelim = idxValueDelim
    val lineDelim = delim
    val d = trueFeatureDimension
    val isZeroBased = zeroBased

    sc.textFile(inputPath, minPartitions)
      .map { case line =>
        val ts = line.split(lineDelim)

        val label = ts(0).toDouble
        val binaryLabel = if (label > 0) 1d else 0d

        val indices = new Array[Int](ts.length - 1)
        val values = new Array[Double](ts.length - 1)
        for (i <- 1 until ts.length) {
          val idxAndVal = ts(i).split(itemDelim)
          indices(i - 1) = idxAndVal(0).toInt
          // Photon ML assumes all features start from zero
          if (!isZeroBased) {
            indices(i - 1) -= 1
          }
          values(i - 1) = idxAndVal(1).toDouble
        }

        val features = new SparseVector[Double](indices, values, d)
        new LabeledPoint(features = features, label = binaryLabel, weight = 1d)
      }
  }

  /**
   *
   * @return IndexMapLoader
   */
  override def indexMapLoader(): IndexMapLoader = _indexMapLoader
}
