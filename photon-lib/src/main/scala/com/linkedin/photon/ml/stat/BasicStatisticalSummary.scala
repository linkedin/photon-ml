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
package com.linkedin.photon.ml.stat

import breeze.linalg.{Vector => BreezeVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.util.{Logging, VectorUtils}

/**
 * Class BasicStatisticalSummary contains various statistics about that data.
 *
 * It is a wrapper around spark-ml MultivariateStatisticalSummary, that relies on breeze Vector rather than
 * spark-ml Vector. We also tweak the numbers for our needs (see calculateBasicStatistics).
 *
 * @note variance is calculated by spark.ml to be unbiased, so based on N-1 degrees of freedom, which is the
 * standard statistical practice. A degree of freedom is lost when using an estimated mean to compute the variance.
 *
 * TODO: rename just "BasicStatistics": descriptive statistics are summaries of the data by definition
 * TODO: need to replace the MultivariateStatisticalSummary with the spark.ml.stat.Summarizer
 * (as part of the changes from spark.mllib to spark.ml)
 */
case class BasicStatisticalSummary(
    mean: BreezeVector[Double],
    variance: BreezeVector[Double],
    count: Long,
    numNonzeros: BreezeVector[Double],
    max: BreezeVector[Double],
    min: BreezeVector[Double],
    normL1: BreezeVector[Double],
    normL2: BreezeVector[Double],
    meanAbs: BreezeVector[Double],
    interceptIndex: Option[Int])

/**
 * Object BasicStatisticalSummary has functions to actually compute statistical summaries from RDDs.
 */
object BasicStatisticalSummary extends Logging {

  /**
   * This function accepts a RDD[LabeledPoint]. Used in Photon.
   *
   * @param inputData The input data (usually training data)
   * @param p A DummyImplicit to allow the Scala compiler to distinguish from other apply function
   * @return An instance of BasicStatisticalSummary
   */
  def apply
      (inputData: RDD[LabeledPoint], interceptIndex: Option[Int])
      (implicit p: DummyImplicit): BasicStatisticalSummary =
    calculateBasicStatistics(
      Statistics.colStats(inputData.map(x => VectorUtils.breezeToMllib(x.features))),
      interceptIndex)

  /**
   * This function accepts a RDD[MLVector]. Used in GAME.
   *
   * @param inputData The input data (usually training data)
   * @return An instance of BasicStatisticalSummary
   */
  def apply(inputData: RDD[MLVector], interceptIndex: Option[Int]): BasicStatisticalSummary =
    calculateBasicStatistics(Statistics.colStats(inputData.map(Vectors.fromML)), interceptIndex)

  /**
   * Auxiliary function to scale vectors.
   *
   * @param count The number to use in the denominator of the scaling
   * @param vector The vector to scale
   * @return The vector, element wise divided by 1/count, or vector if count == 0
   */
  private def scale(count: Long, vector: BreezeVector[Double]): BreezeVector[Double] =
    if (count > 0) vector / count.toDouble else vector

  /**
   * The function that actually calculates the statistics.
   *
   * @param summary An instance of spark-ml MultivariateStatisticalSummary
   * @return An instance of BasicStatisticalSummary
   */
  protected[ml] def calculateBasicStatistics(
      summary: MultivariateStatisticalSummary,
      interceptIndex: Option[Int]): BasicStatisticalSummary = {

    val meanAbs = scale(summary.count, VectorUtils.mllibToBreeze(summary.normL1))
    val tMean = VectorUtils.mllibToBreeze(summary.mean)
    val tVariance = VectorUtils.mllibToBreeze(summary.variance)
    val tNumNonZeros = VectorUtils.mllibToBreeze(summary.numNonzeros)
    val tMax = VectorUtils.mllibToBreeze(summary.max)
    val tMin = VectorUtils.mllibToBreeze(summary.min)
    val tNormL1 = VectorUtils.mllibToBreeze(summary.normL1)
    val tNormL2 = VectorUtils.mllibToBreeze(summary.normL2)

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

    BasicStatisticalSummary(
      tMean,
      tVariance.mapActiveValues(x => if (x.isNaN || x.isInfinite || x < 0) 1.0 else x),
      summary.count,
      tNumNonZeros,
      tMax,
      tMin,
      tNormL1,
      tNormL2,
      meanAbs,
      interceptIndex)
  }
}
