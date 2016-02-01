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
import org.apache.spark.mllib.linalg.VectorsWrapper
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary


/**
 * A wrapper of [[https://spark.apache.org/docs/1.4.0/api/scala/index.html#org.apache.spark.mllib.stat.MultivariateStatisticalSummary MultivariateStatisticalSummary]]
 * of mllib to use breeze vectors instead of mllib vectors.
 * The summary provides mean, variance, max, min, normL1 and normL2 for each features, as well as the expected magnitude of features (meanAbs) to assist in computing
 * feature importance.
 *
 * @author dpeng
 */
case class BasicStatisticalSummary(mean: Vector[Double],
                                   variance: Vector[Double],
                                   count: Long,
                                   numNonzeros: Vector[Double],
                                   max: Vector[Double],
                                   min: Vector[Double],
                                   normL1: Vector[Double],
                                   normL2: Vector[Double],
                                   meanAbs: Vector[Double])

object BasicStatisticalSummary {
  /**
   * Convert a [[https://spark.apache.org/docs/1.4.0/api/scala/index.html#org.apache.spark.mllib.stat.MultivariateStatisticalSummary MultivariateStatisticalSummary]]
   * of mllib to a case instance with breeze vectors.
   *
   * @param mllibSummary Summary from mllib
   * @return The summary with breeze vectors
   */
  def apply(mllibSummary: MultivariateStatisticalSummary, meanAbs: Vector[Double]): BasicStatisticalSummary = {
    val tMean = VectorsWrapper.mllibToBreeze(mllibSummary.mean)
    val tVariance = VectorsWrapper.mllibToBreeze(mllibSummary.variance)
    val tNumNonZeros = VectorsWrapper.mllibToBreeze(mllibSummary.numNonzeros)
    val tMax = VectorsWrapper.mllibToBreeze(mllibSummary.max)
    val tMin = VectorsWrapper.mllibToBreeze(mllibSummary.min)
    val tNormL1 = VectorsWrapper.mllibToBreeze(mllibSummary.normL1)
    val tNormL2 = VectorsWrapper.mllibToBreeze(mllibSummary.normL2)
    this(tMean, tVariance, mllibSummary.count, tNumNonZeros, tMax, tMin, tNormL1, tNormL2, meanAbs)
  }
}
