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
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary

import com.linkedin.photon.ml.util.{Logging, VectorUtils}

/**
 * A wrapper of
 * [[https://spark.apache.org/docs/1.4.0/api/scala/index.html#org.apache.spark.mllib.stat.MultivariateStatisticalSummary
 *  MultivariateStatisticalSummary]]
 * from mllib to use breeze vectors instead of mllib vectors.
 * The summary provides mean, variance, max, min, normL1 and normL2 for each feature, as well as the expected magnitude
 * of features (meanAbs) to assist in computing feature importance.
 */
case class BasicStatisticalSummary(
    mean: Vector[Double],
    variance: Vector[Double],
    count: Long,
    numNonzeros: Vector[Double],
    max: Vector[Double],
    min: Vector[Double],
    normL1: Vector[Double],
    normL2: Vector[Double],
    meanAbs: Vector[Double])

object BasicStatisticalSummary extends Logging {
  /**
   * Converts a
   * [[https://spark.apache.org/docs/1.4.0/api/scala/index.html#org.apache.spark.mllib.stat.
   * MultivariateStatisticalSummary]]
   * of mllib to an instance of case class BasicStatisticalSummary using breeze vectors.
   *
   * @param mllibSummary Summary from mllib
   * @return The summary with breeze vectors
   */
  def apply(mllibSummary: MultivariateStatisticalSummary, meanAbs: Vector[Double]): BasicStatisticalSummary = {


    val tMean = VectorUtils.mllibToBreeze(mllibSummary.mean)
    val tVariance = VectorUtils.mllibToBreeze(mllibSummary.variance)
    val tNumNonZeros = VectorUtils.mllibToBreeze(mllibSummary.numNonzeros)
    val tMax = VectorUtils.mllibToBreeze(mllibSummary.max)
    val tMin = VectorUtils.mllibToBreeze(mllibSummary.min)
    val tNormL1 = VectorUtils.mllibToBreeze(mllibSummary.normL1)
    val tNormL2 = VectorUtils.mllibToBreeze(mllibSummary.normL2)

    val adjustedCount = tVariance.activeIterator.foldLeft(0)((count, idxValuePair) => {
      if (idxValuePair._2.isNaN || idxValuePair._2.isInfinite || idxValuePair._2 < 0) {
        logger.warn(s"Detected invalid variance at index ${idxValuePair._1} (${idxValuePair._2})")
        count + 1
      } else {
        count
      }
    })

    if (adjustedCount > 0) {
      logger.warn(s"Found $adjustedCount features where variance was either non-positive, not-a-number, or " +
        "infinite. The variances for these features have been re-set to 1.0.")
    }

    this(
      tMean,
      tVariance.mapActiveValues(x => if (x.isNaN || x.isInfinite || x < 0) 1.0 else x),
      mllibSummary.count, tNumNonZeros, tMax, tMin, tNormL1, tNormL2, meanAbs)
  }
}
