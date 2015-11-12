/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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

import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.commons.math3.stat.descriptive.rank.PSquarePercentile

/**
 * Miscellaneous things that describe a GLM coefficient.
 *
 * Notes for testing:
 * <ul>
 *   <li>Tests should assume that the Apache Commons stuff is correct -- really just need to establish that
 *   things have been reasonably wired.</li>
 * </ul>
 */
class CoefficientSummary extends Serializable {
  private val summary:SummaryStatistics = new SummaryStatistics()
  private val quantiles:Array[PSquarePercentile] = Array[PSquarePercentile](new PSquarePercentile(0.25),
    new PSquarePercentile(0.50), new PSquarePercentile(0.75))

  def accumulate(x:Double): Unit = {
    summary.addValue(x)
    quantiles.foreach { y => y.increment(x) }
  }

  def getMean(): Double = summary.getMean

  def getMin(): Double = summary.getMin

  def getMax(): Double = summary.getMax

  def getStdDev(): Double = summary.getStandardDeviation

  def estimateFirstQuartile(): Double = quantiles(0).quantile()

  def estimateMedian(): Double = quantiles(1).quantile()

  def estimateThirdQuartile(): Double = quantiles(2).quantile()

  def getCount(): Long = summary.getN

  override def toString():String = {
    f"Range: [Min: $getMin%.03f, Q1: $estimateFirstQuartile%.03f, Med: $estimateMedian%.03f, Q3: $estimateThirdQuartile%.03f, Max: $getMax%.03f) " +
    f"Mean: [$getMean%.03f], Std. Dev.[$getStdDev%.03f], # samples = [$getCount]"
  }
}
