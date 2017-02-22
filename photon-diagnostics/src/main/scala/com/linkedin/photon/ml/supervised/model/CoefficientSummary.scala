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
package com.linkedin.photon.ml.supervised.model

import scala.collection.mutable.ArrayBuffer

import org.apache.commons.math3.stat.descriptive.SummaryStatistics

/**
 * Miscellaneous things that describe a GLM coefficient.
 *
 * Note: this code assumes that a relatively small number of samples (less than thousands) are being provided.
 * If this is not the case, the quantile estimation will need to be revisited.
 */
class CoefficientSummary extends Serializable {
  private val summary: SummaryStatistics = new SummaryStatistics()
  private val quantiles = new ArrayBuffer[Double]()

  def accumulate(x:Double): Unit = {
    summary.addValue(x)
    quantiles += x
  }

  def getMean: Double = summary.getMean

  def getMin: Double = summary.getMin

  def getMax: Double = summary.getMax

  def getStdDev: Double = summary.getStandardDeviation

  def estimateFirstQuartile(): Double = quantiles.sortWith(_ < _)(1 * quantiles.size / 4)

  def estimateMedian(): Double = quantiles.sortWith(_ < _)(2 * quantiles.size / 4)

  def estimateThirdQuartile(): Double = quantiles.sortWith(_ < _)(3 * quantiles.size / 4)

  def getCount: Long = summary.getN

  override def toString:String = {
    f"Range: [Min: $getMin%.03f, Q1: ${estimateFirstQuartile()}%.03f, Med: ${estimateMedian()}%.03f, " +
      f"Q3: ${estimateThirdQuartile()}%.03f, Max: $getMax%.03f) " +
    f"Mean: [$getMean%.03f], Std. Dev.[$getStdDev%.03f], # samples = [$getCount]"
  }
}
