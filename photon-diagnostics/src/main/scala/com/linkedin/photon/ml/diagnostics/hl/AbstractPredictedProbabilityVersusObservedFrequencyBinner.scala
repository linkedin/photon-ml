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
package com.linkedin.photon.ml.diagnostics.hl

import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import org.apache.spark.rdd.RDD

/**
 * Handles most of the details of binning scores in a uniform way. The only missing piece of the puzzle is the number of
 * of bins to use, which is delegated to child instances to compute.
 */
abstract class AbstractPredictedProbabilityVersusObservedFrequencyBinner
  extends PredictedProbabilityVersusObservedFrequencyBinner {

  import AbstractPredictedProbabilityVersusObservedFrequencyBinner._

  /**
   * For downstream implementations to figure out: how to choose the right number of bins given the number of samples
   * and the number of feature dimensions
   *
   * @param numItems
   * @param numDimensions
   * @return
   * Number of uniform-sized histogram bins to use
   */
  def getBinCount(numItems: Long, numDimensions: Int): (String, Int)

  def bin(
      numItems: Long,
      numDimensions: Int,
      observedVExpected: RDD[(Double, Double)]):
        (String, Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]) = {

    val (binMsg, actualBins) = getBinCount(numItems, numDimensions)

    // Compute the actual bin contents, phase 1 (get the observed counts)
    val finalCounts = observedVExpected.mapPartitions(x => {
      val result: Array[PredictedProbabilityVersusObservedFrequencyHistogramBin] = generateInitialBins(actualBins.toInt)

      x.foreach(obs => {
        obs match {
          case (actualLabel: Double, predictedScore: Double) =>
            val binToUpdate = findBin(predictedScore, result)

            if (math.abs(BinaryClassifier.positiveClassLabel - actualLabel) < EPSILON) {
              result(binToUpdate).observedPosCount += 1
            } else {
              result(binToUpdate).observedNegCount += 1
            }
        }
      })

      Some(result).iterator
    }).fold(generateInitialBins(actualBins.toInt))((x, y) => {
      x zip y map (z => z._1.accumulate(z._2))
    })

    (binMsg, finalCounts)
  }
}

object AbstractPredictedProbabilityVersusObservedFrequencyBinner {
  val EPSILON = 1e-12

  /**
   * Helper method to generate initially empty bins
   */
  def generateInitialBins(numBins: Int): Array[PredictedProbabilityVersusObservedFrequencyHistogramBin] = {
    require(numBins > 0,
      s"Requested number of bins must be positive, got [$numBins]")

    (0 until numBins).map(x => {
      val binStart = x / numBins.toDouble
      val binEnd = (x + 1) / numBins.toDouble
      new PredictedProbabilityVersusObservedFrequencyHistogramBin(binStart, binEnd)
    }).toArray
  }

  /**
   * Helper method to do binary search on histogram bins to find the proper bin
   * @param predictedScore
   * The predicted score whose bucket we are attempting to lookup
   * @param bins
   * The bins to search. The assumption is that the bins are already sorted such that:
   * <ul>
   * <li><tt>bins(i+1).lowerBound == bins(i).upperBound</tt></li>
   * <li><tt>bins(0).lowerBound == 0</tt></li>
   * <li><tt>bins(last).upperBound == 1</tt></li>
   * <li><tt>bins(n).lowerBound &lt; bins(n).upperBound</tt></li>
   * </ul>
   * @return
   * The index of the bin that should receive the increment
   */
  def findBin(predictedScore: Double, bins: Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]): Int = {
    require(predictedScore >= 0.0 && predictedScore <= 1.0,
      s"Predicted score [$predictedScore] is in the range [0,1]")

    var minIdx: Int = 0
    var maxIdx: Int = bins.size - 1
    var midIdx: Int = (maxIdx + minIdx) / 2
    var done: Boolean = false

    while (!done) {
      if (maxIdx == minIdx) {
        done = true
      } else if (predictedScore >= bins(midIdx).upperBound) {
        // Above the mid-bin's upper bound
        minIdx = midIdx + 1
      } else if (predictedScore < bins(midIdx).lowerBound) {
        // Below the mid-bin's lower bound
        maxIdx = midIdx - 1
      } else {
        done = true
      }

      if (!done) {
        midIdx = (minIdx + maxIdx) / 2
      }
    }

    midIdx
  }
}
